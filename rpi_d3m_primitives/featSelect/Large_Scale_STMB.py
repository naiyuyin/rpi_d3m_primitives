import numpy as np
from rpi_d3m_primitives.featSelect.tian_STMB_new import tian_STMB_new
from rpi_d3m_primitives.featSelect.helperFunctions import joint
from rpi_d3m_primitives.featSelect.conditionalMI import cmi


def Large_Scale_STMB(data, targets, method, threshold):
    #data is training data without label column
    numfeat = data.shape[1]
    subsize = 10
    count = 0
    Feat = []
    while count*subsize <= numfeat:
#        print(count)
        if (count + 1)*subsize <= numfeat:
            sub_D = data[:, count*subsize : subsize+count*subsize]
            results = tian_STMB_new(sub_D, targets, method, threshold)
            index = results[0] + count*subsize
            Feat = set(Feat).union(set(index))
        else:
            sub_D = data[:, count*subsize :]
            results = tian_STMB_new(sub_D, targets, method, threshold)
            index = results[0] + count*subsize
            Feat = set(Feat).union(set(index))
        count = count + 1
    
    Feat = list(Feat) #convert set object to list
    cmbVector = joint(data[:, Feat])
    for i in np.setdiff1d(np.arange(numfeat), Feat):
        temp = cmi(data[:,i], targets, cmbVector)
        if temp > threshold:
            Feat.append(i)
    
    MB = Feat
    return np.array(MB)