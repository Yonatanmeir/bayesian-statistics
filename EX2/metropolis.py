# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:24:17 2020

@author: Yonatan Meir
"""

import numpy as np

def metropolis_fcn(density_fcn,init,width,num_samples):
    param_steps=[]
    cur_param=init
    param_steps=init
    for i in np.arange(num_samples):
        p_cur=density_fcn(cur_param)
        prop_param=cur_param+np.random.normal(loc=0,scale=width,size=(1,1))
        try:
            if init.shape[0]>1:
                prop_param=cur_param+np.random.multivariate_normal(mean=np.zeros(init.shape[0]),cov=width*np.eye(init.shape[0]),size=(1))
        except:
            a=1
        p_prop=density_fcn(prop_param.transpose())
        try:
            p_move=p_prop/p_cur
            tresh=np.random.uniform(size=(1,1))
            if tresh<=p_move:
                cur_param=prop_param[0]
            param_steps=np.vstack((param_steps,cur_param))
        except:
            param_steps=np.vstack((param_steps,cur_param))
    return param_steps
def gibbs_sampler(init,num_samples,func_mu,func_sigma):
    param_steps=[]
    cur_mu=init[0]
    cur_sigma=init[1]
    mu_steps=init[0]
    sigma_steps=init[1]
    for i in np.arange(num_samples): 
        cur_mu=func_mu(cur_mu,cur_sigma)
        cur_sigma= func_sigma(cur_mu,cur_sigma)    
        mu_steps=np.vstack((mu_steps,cur_mu))
        sigma_steps=np.vstack((sigma_steps,cur_sigma))
    return mu_steps,sigma_steps
