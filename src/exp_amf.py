import numpy as np
from matplotlib.pyplot import *
from matplotlib.colors import TwoSlopeNorm

import time
import sys
import ipdb


def signGD_simulator_simplified(x0_init, x1_init, x2_init, eta = 0.1, steps = 100):

    x0_seq = np.zeros(steps)
    x1_seq = np.zeros(steps)
    x2_seq = np.zeros(steps)
    S_seq = np.zeros(steps)

    x0_seq[0] = x0_init
    x1_seq[0] = x1_init
    x2_seq[0] = x2_init

    for m in range(steps-1):
        
        x0_old = x0_seq[m]
        x0_new = x0_old - eta*np.sqrt(3)*np.sign(x0_old)
        
#        x0_seq[m+1] = x0_seq[m] - eta*np.sqrt(3)*np.sign(x0_seq[m])

        x1_old = x1_seq[m]
        x1_new = x1_old - eta*np.sqrt(2)*np.sign(x1_old)
#        x1_seq[m+1] = x1_seq[m] - eta*np.sqrt(3)*np.sign(x1_seq[m])

        change_indicator = int( abs(x0_old/np.sqrt(3))<abs(x1_old/np.sqrt(2)) )
        S_seq[m] = change_indicator
        
        x2_old = x2_seq[m]
        x2_new = x2_old - eta*np.sqrt(2/3)*change_indicator*np.sign(-x0_old)


        x0_seq[m+1] = x0_new
        x1_seq[m+1] = x1_new
        x2_seq[m+1] = x2_new
        
        S_seq[m] = change_indicator
    

    return x0_seq, x1_seq, x2_seq, S_seq

def signGD_simulator_paper(x0_init, x1_init, x2_init, sigma, eta = 0.1, steps = 100):
    
    sigma_0_squared = sigma[0]
    sigma_1_squared = sigma[1]

    x0_seq = np.zeros(steps)
    x1_seq = np.zeros(steps)
    x2_seq = np.zeros(steps)
    signA_seq = np.zeros(steps)

    x0_seq[0] = x0_init
    x1_seq[0] = x1_init
    x2_seq[0] = x2_init
    
    T = 0
    within_eta = False
    
    for m in range(steps-1):
        x0_old = x0_seq[m]
        x1_old = x1_seq[m]
        x2_old = x2_seq[m]
        
        A = x0_old/np.sqrt(3)*sigma_0_squared
        B = x1_old/np.sqrt(2)*sigma_1_squared
        
        signApB = np.sign(A+B)
        signAmB = np.sign(A-B)
        signA = np.sign(A)
  
        x0_new = x0_old - eta*np.sqrt(1/3)*(signApB + signA + signAmB)
        x1_new = x1_old - eta*np.sqrt(1/2)*(signApB - signAmB)
        x2_new = x2_old - eta*(np.sqrt(1/6)*(signApB + signAmB) - np.sqrt(2/3)*signA)
        
        x0_seq[m+1] = x0_new
        x1_seq[m+1] = x1_new
        x2_seq[m+1] = x2_new
        signA_seq[m+1] = signA
        
        A_new = x0_new/np.sqrt(3)*sigma_0_squared
        if not within_eta and (abs(A_new) < eta):
            within_eta = True
            T = m+1
        
    return x0_seq, x1_seq, x2_seq, signA_seq, T

def sweep_x0x1_init(signGD_simulator, x0_init_range, x1_init_range, sigma, eta = 0.1, steps = 100):
    
    x0_final = np.zeros( (len(x0_init_range),len(x1_init_range)) )   
    x1_final = np.zeros( (len(x0_init_range),len(x1_init_range)) )   
    x2_final = np.zeros( (len(x0_init_range),len(x1_init_range)) ) 
    
    x0_at_T = np.zeros( (len(x0_init_range),len(x1_init_range)))
    x1_at_T = np.zeros( (len(x0_init_range),len(x1_init_range)))
    x2_at_T = np.zeros( (len(x0_init_range),len(x1_init_range)))
    
    x2_change_after_T = np.zeros( (len(x0_init_range),len(x1_init_range)))
    
    T_final = np.zeros( (len(x0_init_range),len(x1_init_range)))   

    signDiff = np.zeros( (len(x0_init_range),len(x1_init_range))) 


    for (ind0, x0_init) in enumerate(x0_init_range):
        for (ind1, x1_init) in enumerate(x1_init_range):
            
            x0_seq, x1_seq, x2_seq, signA_seq, T = signGD_simulator(
                x0_init=x0_init, x1_init=x1_init, x2_init=0., sigma=sigma, eta=eta, steps=steps)
            x0_final[ind0,ind1] = np.mean(x0_seq[-10:])
            x1_final[ind0,ind1] = np.mean(x1_seq[-10:])
            x2_final[ind0,ind1] = np.mean(x2_seq[-10:]) # smoothing the final value
#             S_sign_change_sum[ind0,ind1] = sum(S_seq*np.sign(x0_seq)) 
            
            x0_at_T[ind0,ind1] = x0_seq[T]
            x1_at_T[ind0,ind1] = x1_seq[T]
            x2_at_T[ind0,ind1] = x2_seq[T]
            
            x2_change_after_T[ind0,ind1] = x2_seq[-1] - x2_seq[T+1]
#             ipdb.set_trace()
            signDiff[ind0,ind1] = (signA_seq[T+1:]==1).sum() - (signA_seq[T+1:]==-1).sum()

            T_final[ind0,ind1] = T

    return x0_final, x1_final, x2_final, x0_at_T, x1_at_T, x2_at_T, x2_change_after_T, T_final, signDiff