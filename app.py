import numpy as np
# import pyximport
# pyximport.install()
from intuition_fuzzy import IntuitiveFuzzy
#import IntuitiveFuzzy
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

data_file = 'heart.csv'
#att_nominal_cate = ["a1","a3","a4","a6","a7","a9","a10","a12","a14","a15","a17","a19","a20","d"]
att_nominal_cate = ["d"]
min_max_scaler = preprocessing.MinMaxScaler()


def preprocessing(pathfile):
    DS  = np.genfromtxt(pathfile, delimiter=",", dtype=object)[:,:]
    att = DS[0].astype(str)
    att_real = [i for i in att  if i not in att_nominal_cate ]

    DS[0] = att
    list_index_cate = [list(DS[0]).index(i) for i in att_nominal_cate]
    for i in list_index_cate:
        DS[1:, i] = LabelEncoder().fit_transform(DS[1:,i])

    DS[1:,:] = DS[1:,:].astype(float)
    if len(att_real) > 0 :
        list_index_real = [list(DS[0]).index(i) for i in att_real]
        DS[1:,list_index_real] = min_max_scaler.fit_transform(DS[1:,list_index_real])
    
    return DS

if __name__ == "__main__":
    DS = preprocessing(data_file)
    F = IntuitiveFuzzy(DS, att_nominal_cate)
    W = F.filter(verbose=True)
    print(W)
    print(F.evaluate(W,10))
