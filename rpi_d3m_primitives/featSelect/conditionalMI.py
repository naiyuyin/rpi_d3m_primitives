import numpy as np
from rpi_d3m_primitives.featSelect.helperFunctions import normalize_array, joint
from rpi_d3m_primitives.featSelect.mutualInformation import mi, joint_probability
from rpi_d3m_primitives.featSelect.mutualInformation import MutualInfo_funs, mi_pseudoBayesian, mi_Bayesian, mi_Full_Bayesian_fixPt

"""---------------------------- CONDITIONAL MUTUAL INFORMATION ----------------------------"""
def mergeArrays(firstVector, secondVector, length):
    
    if length == 0:
        length = firstVector.size
    
    results = normalize_array(firstVector, 0)
    firstNumStates = results[0]
    firstNormalisedVector = results[1]
    
    results = normalize_array(secondVector, 0)
    secondNumStates = results[0]
    secondNormalisedVector = results[1]
    
    stateCount = 1
    stateMap = np.zeros(shape = (firstNumStates*secondNumStates,))
    merge = np.zeros(shape =(length,))

    joint_states = np.column_stack((firstVector,secondVector))
    uniques,merge = np.unique(joint_states,axis=0,return_inverse=True)
    stateCount = len(uniques)
    results = []
    results.append(stateCount)
    results.append(merge)
    return results


def conditional_entropy(dataVector, conditionVector, length):
    condEntropy = 0
    jointValue = 0
    condValue = 0
    if length == 0:
        length = dataVector.size
    
    results = joint_probability(dataVector, conditionVector, 0)
    jointProbabilityVector = results[0]
    numJointStates = results[1]
    numFirstStates = results[3]
    secondProbabilityVector = results[4]
    
    for i in range(0, numJointStates):
        jointValue = jointProbabilityVector[i]
        condValue = secondProbabilityVector[int(i / numFirstStates)]
        if jointValue > 0 and condValue > 0:
            condEntropy -= jointValue * np.log2(jointValue / condValue);

    return condEntropy


def cmi(dataVector, targetVector, conditionVector, length = 0):
    if (conditionVector.size == 0):
        return mi(dataVector,targetVector,0)
    if (len(conditionVector.shape)>1 and conditionVector.shape[1]>1):
        conditionVector = joint(conditionVector)
    cmi = 0;
    firstCondition = 0
    secondCondition = 0
    
    if length == 0:
        length = dataVector.size
    
    results = mergeArrays(targetVector, conditionVector, length)
    mergedVector = results[1]
    
    firstCondition = conditional_entropy(dataVector, conditionVector, length)
    secondCondition = conditional_entropy(dataVector, mergedVector, length)
    cmi = firstCondition - secondCondition
    
    return cmi

###############################################################################################
#created 05/29/2019
def ConditionalMutualInfo_funs(X, Y, Z, method, *args):
	if method == 'counting':
		mutu_info = cmi(X,Y,Z)
		return mutu_info

	mutu_info = 0
	alpha = 1

	if len(args) == 0:
		m = len(np.unique(X))
		n = len(np.unique(Y))
		k = len(np.unique(Z))
	else:
		m = args[0]
		n = args[1]
		k = args[2]

	if Z.size == 0:
		mutu_info = MutualInfo_funs( X, Y, method, m, n)
	else:
		hm_samples = X.shape[0]
#		if Z.shape[1] != 1:
#			Z = joint(Z)
		states = np.unique(Z)
		for i in range(len(states)):
			pattern = states[i]
			sub_cond_idx = np.where(Z == pattern)
			tt = sub_cond_idx[0].shape[0]
			if method == 'pseudoBayesian':
				temp_mi = mi_pseudoBayesian(X[sub_cond_idx[0]], Y[sub_cond_idx[0]], m, n)
				mutu_info += (tt + alpha)*temp_mi/(hm_samples + alpha*k) 
			elif method == 'Bayesian':
				temp_mi = mi_Bayesian(X[sub_cond_idx[0]], Y[sub_cond_idx[0]], m, n, alpha)
				mutu_info += tt*temp_mi/hm_samples
			elif method == 'fullBayesian':
				temp_mi,_ = mi_Full_Bayesian_fixPt(X[sub_cond_idx[0]], Y[sub_cond_idx[0]], m, n)
				mutu_info += tt*temp_mi/hm_samples
			sub_cond_idx = None
			temp_mi = None
	return mutu_info

