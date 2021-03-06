""" Read Page 3 """
import numpy as np
import time

from tabulate import tabulate
from sklearn import svm
from sklearn.model_selection import cross_val_score

class IntuitiveFuzzy(object):

	
	def __init__(self, dataframe, att_nominal_cate):
		super(IntuitiveFuzzy, self).__init__()
		#assert isinstance(dataframe, pd.DataFrame)

		self.lambda_ = 1
		self.ro = 0.7

		print('[INFO] Initializing object ...')
		self.data = dataframe
		
		# Including decision. Assume last column is decision values
		self.attributes = list(dataframe[0])
		self.C = self.attributes[:-1]
		self.arr_cate = att_nominal_cate
		self.arr_real = [i for i in self.attributes  if i not in att_nominal_cate ]
		
		### For filtering phase ###
		self.num_attr = len(self.data[0])
		self.num_objs = len(self.data[1:])
		self.relational_matrices = self._get_single_attr_IFRM(self.data)


	def __str__(self):
		string = f"Attributes list : {str(self.attributes)}\n\n"
	
		for attr in self.attributes:
			string+=f'Relational matrix of {attr} : \n{str(self.relational_matrices[attr])}\n\n'

		return string


	def _get_single_attr_IFRM(self, data):
		"""
			This function returns a dictionary of relational matrices corresponding to
			each single attributes

			Params :
				- data : The pandas DataFrame of sample data 

			Returns : 
				- result : The dictionary of relational matrices corresponding each attribute 
		"""
		result = {}
		list_index_real = [list(self.attributes).index(i) for i in self.arr_real] 
		for k in range(len(self.attributes)):
			column = data[1:,k]
			rel_matrix = np.empty((self.num_objs, self.num_objs), dtype=tuple)

			if k in list_index_real:
				for i in range(self.num_objs):
					for j in range(self.num_objs):

						mu = round(1 - abs(column[i] - column[j]), 2)
						v  = round((1 - mu) / (1 + self.lambda_ * mu), 2)

						rel_matrix[i][j] = (mu, v)
			
			else:
				for i in range(self.num_objs):
					for j in range(self.num_objs):

						if column[i] == column[j]:
							mu = 1.0
							v  = round((1 - mu) / (1 + self.lambda_ * mu), 2)
						else:
							mu = 0
							v  = round((1 - mu) / (1 + self.lambda_ * mu), 2)		
						rel_matrix[i][j] = (mu, v)

			result[self.attributes[k]] = rel_matrix

		return result
		

	def _intersect_ifr(self, tup_list):
		"""
			This function complements _get_multiple_attr_IFRM, it returns
			IFR(Q Union P) of the current tuple list 

			Params :
				- tup_list : A tuple list in the form (mu, v)

			Returns :
				- result : a tuple in the form of (inf(mu), sup(v))
		"""

		return (min(_x[0] for _x in tup_list), max(_x[1] for _x in tup_list))

	def _get_multiple_attr_IFRM(self, attributes):
		"""
			This function returns the intuitive relational matrix of two or more attributes

			Params :
				- attributes : List of attributes 

			Returns :
				- result : The relational matrix of the attributes partition
		"""
		assert len(attributes) >= 1
		if(len(attributes) == 1): return self.relational_matrices[attributes[0]]

		combined = np.empty((len(attributes), self.num_objs, self.num_objs), dtype=tuple)
		for i, attr in enumerate(attributes):
			combined[i,:,:] = self.relational_matrices[attr]

		combined = np.apply_along_axis(self._intersect_ifr, 0, combined)
		result = np.empty((self.num_objs, self.num_objs), dtype=tuple)

		for i in range(self.num_objs):
			for j in range(self.num_objs):
				result[i][j] = (combined[0, i, j], combined[1, i, j])

		return result

	def _get_intersect_IFRM(self, IFRM_1, IFRM_2):
		"""
			This function will return the intuitive  relational matrix of P intersect Q
			where P and Q are two sets of attributes which are both complete subset of C.
			Note : in the paper R{P intersect Q} = R{P} union R{Q}

			Params :
				- p1 : First subset of attributes
				- p2 : Second subset of attributes

			Returns :
				- result : The IFRM of P intersect Q
		"""
		result = np.empty((self.num_objs, self.num_objs), dtype=tuple)

		for i in range(self.num_objs):
			for j in range(self.num_objs):
				result[i, j] = (max(IFRM_1[i,j][0], IFRM_2[i,j][0]), min(IFRM_1[i, j][1], IFRM_2[i, j][1]))

		return result	

	def _get_union_IFRM(self, IFRM_1, IFRM_2):
		"""
			This function will return the intuitive  relational matrix of P union Q
			where P and Q are two sets of attributes which are both complete subset of C.
			Note : in the paper R{P union Q} = R{P} intersect R{Q}

			Params :
				- p1 : First subset of attributes
				- p2 : Second subset of attributes

			Returns :
				- result : The IFRM of P intersect Q
		"""
		result = np.empty((self.num_objs, self.num_objs), dtype=tuple)

		for i in range(self.num_objs):
			for j in range(self.num_objs):
				result[i, j] = (min(IFRM_1[i,j][0], IFRM_2[i,j][0]), max(IFRM_1[i, j][1], IFRM_2[i, j][1]))

		return result

	def _get_cardinality(self, IFRM):
		"""
			Returns the caridnality of a partition of attributes 

			Params :
				- IFRM : An intuitive fuzzy relational matrix

			Returns :
				- caridnality : The caridnality of that parition 
		"""
		caridnality = 0
		for i in range(self.num_objs):
			for j in range(self.num_objs):
				mu = IFRM[i, j][0]
				v  = IFRM[i, j][1]

				caridnality += round(((1 + mu - v) / 2),2)
		return caridnality

	def intuitive_partition_dist(self, p1, p2):
		"""
			This function returns the distance between two partitions of attributes. 
			Note : When calculating the distance between an empty partition and a non-empty
			partition. Just take the cardinality of the non-empty parition multiplied by |U|**2
			This function use in step 1.

			Params :
				- p1 : First partition of attributes 
				- p2 : Second partition of attributes 

			Returns :
				- result : A scalar representing the distance
		"""
		if (len(p1) == 0):
			#IFRM = self._get_multiple_attr_IFRM(p2)
			#return round(self._get_cardinality(IFRM) * (1/(self.num_objs*self.num_objs)),2)
			return 1
		else:
			IFRM_1 = self._get_multiple_attr_IFRM(p1)
			return self.intuitive_partition_dist_d(IFRM_1)


	def sig_start(self, B, a):
		"""
			This function measures the significance of an attribute a to the set of 
			attributes B. This function use step 1

			Params :
				- B : list of attributes 
				- a : an attribute in C but not in B

			Returns :
				- sig : significance value of a to B
		"""
		assert isinstance(B, list)
		assert a not in B and a in self.C

		sig = 0

		d1 = self.intuitive_partition_dist(B, B + [self.attributes[-1]])
		d2 = self.intuitive_partition_dist(B + [a], B + [self.attributes[-1]] + [a])
		sig = round((d1 - d2),2)

		return sig	
	

	def intuitive_partition_dist_d(self, IFRM):
		"""
			This function returns the distance partition to d. 

			Params : IFRM is intuitiononstic fuzzy relation matrix 

			Returns :
				- result : A scalar representing the distance
		"""
		IFRM_cardinality = self._get_cardinality(IFRM)
		IFRM_d = self._get_union_IFRM(IFRM,self.relational_matrices[self.attributes[-1]])
		IFRM_d_cardinality = self._get_cardinality(IFRM_d)
		return round((1 / ((self.num_objs)**2)) * (IFRM_cardinality - IFRM_d_cardinality),2)


	def sig(self, IFRM, a):
		"""
			This function measures the significance of an attribute a to the set of 
			attributes B. This function begin use step 2.

			Params :
				- B : list of attributes 
				- a : an attribute in C but not in B

			Returns :
				- sig : significance value of a to B
		"""
		sig = 0
		print(a)
		d1 = self.intuitive_partition_dist_d(IFRM)
		d2 = self.intuitive_partition_dist_d(self._get_union_IFRM(IFRM,self.relational_matrices[a]))

		sig = round((d1 - d2),2)

		return sig
	

	def condition_stop (self, IFRM_1, IFRM_2):
		IFRM_1_d = self._get_union_IFRM(IFRM_1, self.relational_matrices[self.attributes[-1]])
		IFRM_2_d = self._get_union_IFRM(IFRM_2, self.relational_matrices[self.attributes[-1]])
		IFRM_condition = np.empty((self.num_objs, self.num_objs), dtype=tuple)

		for i in range(self.num_objs):
			for j in range(self.num_objs):
				IFRM_condition[i, j] = (abs(IFRM_1_d[i,j][0]-IFRM_2_d[i,j][0]), abs(IFRM_1_d[i, j][1] - IFRM_2_d[i, j][1]))
		sup_m = np.max(np.max(IFRM_condition,0),0)[0]
		suy_v = np.max(np.max(IFRM_condition,0),0)[1]
		return max(sup_m,suy_v)

	def filter(self, verbose=False):
		"""
			The main function for the filter phase

			Params :
				- verbose : Show steps or not

			Returns :
				- W : A list of potential attributes list
		"""
		# initialization 
		start = time.time()
		B = []
		W = []
		d = self.intuitive_partition_dist(B, B + [self.attributes[-1]])
		D = self.intuitive_partition_dist(self.C, self.attributes)		
		matrix_C = self._get_multiple_attr_IFRM(self.C)

		if(verbose):
			print('\n----- Filtering phase -----')
			print('[INFO] Initialization for filter phase done ...')
			print('    --> Distance from B --> (B union {d}) : %.2f' % d)
			print('    --> Distance from C --> (C union {d}) : %.2f' % D)
			print('-------------------------------------------------------')

		# Filter phase 
		num_steps = 1
		max_sig = 0
		c_m = None

		for c in (self.C):
			SIG_B_c = self.sig_start(B, c)
			#print(c,SIG_B_c)
			if(SIG_B_c > max_sig):
					max_sig = SIG_B_c
					c_m = c

		B.append(c_m)
		W.append(B.copy())
		if(verbose):
			print(f'[INFO] Step {num_steps} completed : ')
			print(f'    --> Max(SIG_B_c) : {round(max_sig,3)}')
			print(f'    --> Selected c_m = {c_m}')
			print(f'    --> Distance from B -> (B union d) : {d}\n')
		IFRM_TG = self._get_multiple_attr_IFRM(B)
		condition = self.condition_stop(IFRM_TG, matrix_C)

		while (condition >= 1 - self.ro and d > D):
			max_sig = 0
			c_m = None

			#for c in set(self.C).difference(set(B)):
			for c in np.setdiff1d(self.C,B):	
				SIG_B_c = self.sig(IFRM_TG, c)
				if(SIG_B_c >= max_sig):
					max_sig = SIG_B_c
					c_m = c

			IFRM_TG = self._get_union_IFRM(IFRM_TG,self.relational_matrices[c_m])
			B.append(c_m)
			W.append(B.copy()) 

			# Re-calculate d
			d = self.intuitive_partition_dist_d(IFRM_TG)
			condition = self.condition_stop(IFRM_TG, matrix_C)

			if(verbose):
				print(f'[INFO] Step {num_steps + 1} completed : ')
				print(f'    --> Max(SIG_B_c) : {round(max_sig,2)}')
				print(f'    --> Selected c_m = {c_m}')
				print(f'    --> Distance from B -> (B union d) : {d}\n')

			# increase step number
			num_steps += 1
		finish = time.time() - start
		print("time process:",finish)
		return W
	
	def evaluate(self, reduct, k=2):

		y_train = self.data[1:,-1]
		y_train = y_train.astype(int)
		X_train_original = self.data[1:,:-1]
		st_org = time.time()
		clf = svm.SVC(kernel='linear', C=1, random_state=42)
		scores_original = round(cross_val_score(clf, X_train_original, y_train, cv=k).mean(),3)
		fn_org = round(time.time() - st_org,3)


		attribute_reduct = reduct[-1]
		list_index = [list(self.data[0,:-1]).index(i) for i in attribute_reduct]

		X_train = self.data[1:,list_index]
		st_reduct = time.time() 
		clf = svm.SVC(kernel='linear', C=1, random_state=42)
		scores = round(cross_val_score(clf, X_train, y_train, cv=k).mean(),3)
		fn_reduct = round(time.time() - st_reduct,3)
		head = ["Size-O", "Size-R", "T-O", "T-R", "Acc-O", "Acc-R"]
		my_data = [[len(self.attributes)-1,len(reduct[-1]),fn_org,fn_reduct,scores_original,scores]]
		return tabulate(my_data, headers=head, tablefmt="grid")
		
