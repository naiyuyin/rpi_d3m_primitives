import numpy as np

from rpi_d3m_primitives.featSelect.helperFunctions import find_probs, normalize_array

from sklearn.metrics import mutual_info_score



def joint_probability(firstVector, secondVector, length):

    if length == 0:

         length = firstVector.size

    

    results = normaliseArray(firstVector, 0)

    firstNumStates = results[0]

    firstNormalisedVector = results[1]

    

    results = normaliseArray(secondVector, 0)

    secondNumStates = results[0]

    secondNormalisedVector = results[1]

    

    jointNumStates = firstNumStates * secondNumStates

    

    firstStateProbs = find_probs(firstNormalisedVector)

    secondStateProbs = find_probs(secondNormalisedVector)

    jointStateProbs = np.zeros(shape = (jointNumStates,))



    # Joint probabilities

    jointStates = np.column_stack((firstNormalisedVector,secondNormalisedVector))

    jointIndices,jointCounts = np.unique(jointStates,axis=0, return_counts = True)

    jointIndices = jointIndices.T

    jointIndices = jointIndices[1]*firstNumStates + jointIndices[0]

    jointIndices = jointIndices.astype(int)

    jointStateProbs[jointIndices] = jointCounts

    jointStateProbs /= length

    

    results = []

    results.append(jointStateProbs)

    results.append(jointNumStates)

    results.append(firstStateProbs)

    results.append(firstNumStates)

    results.append(secondStateProbs)

    results.append(secondNumStates)

    return results





def mi(dataVector, targetVector, length = 0):

    dataVector = dataVector.ravel()

    targetVector = targetVector.ravel()

    mi = mutual_info_score(dataVector,targetVector)/np.log(2)

    return mi





#############################################################################################
#create 05/28/2019
def mi_pseudoBayesian(X, Y, *args):

    if len(args) == 0:

        hm_x = len(np.unique(X))

        hm_y = len(np.unique(Y))

        alpha = 1

    elif len(args) == 2:

        hm_x = args[0]

        hm_y = args[1]

        alpha = 1

    else:

        hm_x = args[0]

        hm_y = args[1]

        alpha = args[2]



    hm_samples = np.size(X)

    joint_table = np.zeros((hm_x, hm_y)) + alpha

    joint_table = Crosstab_Parse(X,Y,joint_table)

    joint_table = joint_table / (hm_samples + hm_x*hm_y*alpha)

    px = np.tile(np.sum(joint_table,1), (hm_y, 1)).reshape(hm_x,hm_y)

    py = np.tile(np.sum(joint_table,0), (hm_x, 1)).reshape(hm_x,hm_y)

    print(px)

    

    mutual_info = np.sum(np.sum(joint_table*(np.log2(joint_table)-np.log2(px)-np.log2(py))))



    return mutual_info, joint_table





def mi_Bayesian( X, Y, *args):

    if len(args) == 0:

        hm_x = len(np.unique(X))

        hm_y = len(np.unique(Y))

        alpha = 1

    elif len(args) == 2:

        hm_x = args[0]

        hm_y = args[1]

        alpha = 1

    else:

        hm_x = args[0]

        hm_y = args[1]

        alpha = args[2]



    hm_samples = np.size(X)



    Nxy = np.zeros((hm_x, hm_y))

    Nxy = Crosstab_Parse(X,Y,Nxy)



    Nx = np.tile(np.sum(Nxy, 1), (hm_y, 1)).reshape(hm_x, hm_y)

    Ny = np.tile(np.sum(Nxy, 0), (hm_x, 1)).reshape(hm_x, hm_y)



    temp = psi(Nx + alpha*hm_y + 1) + psi(Ny + alpha*hm_x +1)-psi(Nxy + alpha + 1)

    temp = ((Nxy + alpha)*temp) / (hm_samples + alpha*hm_x*hm_y)



    mutual_info = psi(hm_samples + hm_x*hm_y*alpha + 1) - np.sum(np.sum(temp))



    mutual_info = mutual_info / np.log(2)

    return mutual_info



def mi_MAP_Bayesian_fixPt(X, Y, *args):

    if len(args) == 0:

        hm_x = len(np.unique(X))

        hm_y = len(np.unique(Y))

    else:

        hm_x = args[0]

        hm_y = args[1]



    N = np.size(X)



    K = hm_x*hm_y



    Nxy = np.zeros((hm_x, hm_y))

    Nxy = Crosstab_Parse(X, Y, Nxy)



    Nxy_arr = Nxy.reshape(hm_x*hm_y,1)



    myfun = lambda alpha : alpha * (np.sum(psi(Nxy_arr + alpha)) / K - psi(alpha)) / (psi(alpha*K + N)-psi(alpha*K))

    alpha_star, itera = fixed_point(myfun, 1, 1e-4, 50)



    mutual_info = mi_Bayesian( X, Y, hm_x, hm_y, alpha_star)



    return mutual_info, alpha_star



#############################################################################################







