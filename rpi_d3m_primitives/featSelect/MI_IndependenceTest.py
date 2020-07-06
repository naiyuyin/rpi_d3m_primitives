import numpy as np
import pandas as pd
from scipy.special import gammaln, psi
from rpi_d3m_primitives.featSelect.mutualInformation import MutualInfo_funs
from rpi_d3m_primitives.featSelect.helperFunctions import joint
from rpi_d3m_primitives.featSelect.conditionalMI import ConditionalMutualInfo_funs

def MI_IndepTest(X, Y, method, ThresFunc):
	m = len(np.unique(X))
	n = len(np.unique(Y))
    
	MI = MutualInfo_funs(X, Y, method, m, n)
#	if method == 'counting':
#		c_X = len(np.unique(X))
#		c_Y = len(np.unique(Y))
#		hm_samples = np.size(X) / (c_X*c_Y)
#		THRES = ThresFunc(max(c_X*c_Y, 4), max(hm_samples, 0.2))
#	else:
	
	hm_samples = np.size(X) / (m*n)
	THRES = ThresFunc(m*n, hm_samples)
#	THRES = ThresFunc(min(max(m*n, 4), 100), min(max(hm_samples, 0.2), 100))
	if np.isnan(THRES) and MI < 0.002:
		Independence = 1
#		print("THRES is nan!\n")
	elif not np.isnan(THRES) and MI <= THRES:
		Independence = 1
	else:
		Independence = 0
	return Independence


def CMI_IndepTest(X, Y, Z, method, ThresFunc):
	if Z.size == 0: #or Z.shape[1] == 0:
		Independence = MI_IndepTest(X, Y, method, ThresFunc)
		return Independence
    
	hm_samples = X.shape[0]
    
#	if len(Z.shape) != 0:
#		if Z.shape[1] != 1:
#		   Z = joint(Z)
	
	m = len(np.unique(X))
	n = len(np.unique(Y))
	k = len(np.unique(Z))
	CMI = ConditionalMutualInfo_funs(X, Y, Z, method, m, n, k)
#	if method == 'counting':
#		c_X = len(np.unique(X))
#		c_Y = len(np.unique(Y))
#		samples_per_config = hm_samples / (c_X*c_Y*states)
#		THRES = ThresFunc(max(c_X*c_Y, 4), max(samples_per_config, 0.2))
#	else:
	samples_per_config = hm_samples / (m*n*k)
	THRES = ThresFunc(m*n, samples_per_config)
#	THRES = ThresFunc(min(max(m*n, 4), 100), min(max(samples_per_config, 0.2), 100))
#	THRES = ThresFunc(max(m*n,4), max(samples_per_config, 0.2))
#	print("THRES : %s"%THRES)
	if np.isnan(THRES) and CMI < 0.002:
		Independence = 1
#		print("THRES is nan!\n")
	elif not np.isnan(THRES) and CMI <= THRES:
		Independence = 1
	else:
		Independence = 0

	return Independence
