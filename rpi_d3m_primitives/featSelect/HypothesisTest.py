import numpy as np
from scipy.stats import chi2
from itertools import combinations
import pandas as pd
from scipy.special import gammaln
from rpi_d3m_primitives.featSelect.mutualInformation import mi
from rpi_d3m_primitives.featSelect.conditionalMI import cmi
from rpi_d3m_primitives.featSelect.helperFunctions import joint, Crosstab_Parse


def GTest_I(X, Y):
    sig_level_indep = 0.05
    hm_x = len(np.unique(X))
    hm_y = len(np.unique(Y))
    
    hm_samples = X.size
    g = 2*hm_samples*mi(X,Y,0)
    
    p_val  = 1 - chi2.cdf(g, (hm_x-1)*(hm_y-1))
    
    if p_val < sig_level_indep:
        Independency = 0  # reject the Null-hypothesis
    else:
        Independency = 1
        
    return Independency

def GTest_CI(X,Y,Z):
    g = 0
    sig_level_indep = 0.05
    
    hm_x = len(np.unique(X))
    hm_y = len(np.unique(Y))
    hm_z = len(np.unique(Z))
    
    hm_samples = X.size
    
    if Z.size == 0:
        return GTest_I(X,Y)
    else:
#        if (len(Z.shape)>1 and Z.shape[1]>1):
#            Z = joint(Z)
#        states = np.unique(Z)
#        for i in states:
#            pattern = i
#            sub_cond_idx = np.where(Z == pattern)
#            temp_mi = mi(X[sub_cond_idx], Y[sub_cond_idx],0)
#            g = g + sub_cond_idx.length*temp_mi
        g = 2*hm_samples*cmi(X,Y,Z)
        p_val = 1 - chi2.cdf(g, (hm_x-1)*(hm_y-1)*hm_z)
        
        if p_val < sig_level_indep:
            Independency = 0  # reject the Null-hypothesis
        else:
            Independency = 1
        
    return Independency



################################################################################
#Create 05/28/2019 
#Change on 01/08/2020
def Bayesian_Factor(X, Y, bayesfactor, *args):
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

    
    hm_samples = len(X)

    Nxy = np.zeros((hm_x, hm_y))
    Nxy = Crosstab_Parse(X,Y, Nxy)

    Nx = np.sum(Nxy,1)
    Ny = np.sum(Nxy,0)

    gammaln_x = np.sum(gammaln(alpha + Nx))
    gammaln_y = np.sum(gammaln(alpha + Ny))

    eta = np.arange(hm_samples)
    ln_H0 = gammaln_y + gammaln_x - np.sum(np.log(alpha* hm_x + eta))-np.sum(np.log(alpha*hm_y + eta)) - (hm_x + hm_y) * gammaln(alpha)

    gammaln_xy = np.sum(np.sum(gammaln(alpha + Nxy)))
    ln_H1 = gammaln_xy - np.sum(np.log(alpha* hm_x*hm_y + eta)) - (hm_x*hm_y)*gammaln(alpha)

    ln_K = ln_H1 - ln_H0
    # print("The current bayes factor is %f\n"%bayesfactor)
    # if (2* ln_K) > 0:
    if (2*ln_K) > bayesfactor:
        Independence = 0 # Reject Null Hypothesis
    else:
        Independence = 1

    return Independence


def Bayesian_Factor_conditional( X, Y, Z, bayesfactor, *args):
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

    if Z.size == 0: #or Z.shape[1] == 0:
        # Independence = Bayesian_Factor( X, Y, hm_x, hm_y)
        Independence = Bayesian_Factor( X, Y, bayesfactor, hm_x, hm_y)
    else:
#        if Z.shape[1] != 1:
#            Z = joint(Z) 
        states = np.unique(Z)
        Independence = 1
        for i in range(len(states)):
            pattern = states[i]
            sub_cond_idx = np.where(Z == pattern)
            tt = sub_cond_idx[0].shape[0]
            # Indep = Bayesian_Factor(X[sub_cond_idx[0]], Y[sub_cond_idx[0]], hm_x, hm_y, alpha)
            Indep = Bayesian_Factor(X[sub_cond_idx[0]], Y[sub_cond_idx[0]], bayesfactor, hm_x, hm_y, alpha)
            Independence *= Indep 
    return Independence
#######################################################################################################





