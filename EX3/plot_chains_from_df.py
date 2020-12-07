# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:43:47 2020

@author: Yonatan Meir
"""
import pystan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from hpd import hpd_grid
from scipy.stats import mode
def plot_mcmc(dict_chains,param,path,name2fig,rope=0,ess=0):
    plt.close()
    try:
        if ess==0:
            ok=1
    except:
        cur_ess=ess[param]
    fig=plt.figure(figsize=(10,10))
    ax_iterations=fig.add_subplot(211)
    ax_histogram=fig.add_subplot(212)
    colors = cm.rainbow(np.linspace(0, 1, len(dict_chains)))
    for counter,c in zip(dict_chains,colors):
        cur_chain=dict_chains[str(counter)][param]
        try:
            if ess==0:
                ax_iterations.plot(np.arange(1,dict_chains['1'].shape[0]+1),cur_chain,color=c,label='chain '+str(counter),alpha=0.2)
        except:
                ax_iterations.plot(np.arange(1,dict_chains['1'].shape[0]+1),cur_chain,color=c,label='chain '+str(counter)+' ess='+str(round(cur_ess[round(int(counter))-1])),alpha=0.2)
        ax_histogram.hist(cur_chain,bins=100,color=c,label='chain '+str(counter),alpha=0.5)
    a=hpd_grid(cur_chain)
    l=a[0][0][0]
    r=a[0][0][1]
    try:
        if  rope== 0:
            ax_histogram.plot(np.array([l,l]),np.array([0,np.histogram(cur_chain,bins=100)[0].max()]),color='k',linestyle='dashed',label='hdi=['+str(l)+'  '+str(r)+']')
            ax_histogram.plot(np.array([r,r]),np.array([0,np.histogram(cur_chain,bins=100)[0].max()]),linestyle='dashed',color='k')
            ax_histogram.text((l+r)/2,np.histogram(cur_chain,bins=100)[0].max(),'mode='+str(round(mode(cur_chain)[0][0],2)))
            ax_iterations.set_xlabel('iteration')
            ax_iterations.set_ylabel('value')
            ax_iterations.set_title(param)
            ax_histogram.set_xlabel('value')
            ax_histogram.set_ylabel('counts')
            ax_iterations.legend(loc='upper right')
            ax_histogram.legend(loc='upper right')
            fig.savefig(path+'\\'+ name2fig)
    except: 
         ax_histogram.plot(np.array([l,l]),np.array([0,np.histogram(cur_chain,bins=100)[0].max()]),color='k',linestyle='dashed',label='hdi=['+str(l)+'  '+str(r)+']')
         ax_histogram.plot(np.array([r,r]),np.array([0,np.histogram(cur_chain,bins=100)[0].max()]),linestyle='dashed',color='k')
         ax_histogram.plot(np.array([rope[0],rope[0]]),np.array([0,np.histogram(cur_chain,bins=100)[0].max()]),color='r',linestyle='dashed',label='rope=['+str(rope[0])+'  '+str(rope[1])+']')
         ax_histogram.plot(np.array([rope[1],rope[1]]),np.array([0,np.histogram(cur_chain,bins=100)[0].max()]),linestyle='dashed',color='r')
         ax_histogram.text((l+r)/2,np.histogram(cur_chain,bins=100)[0].max(),'mode='+str(round(mode(cur_chain)[0][0],2)))
         ax_iterations.set_xlabel('iteration')
         ax_iterations.set_ylabel('value')
         ax_iterations.set_title(param)
         ax_histogram.set_xlabel('value')
         ax_histogram.set_ylabel('counts')
         ax_iterations.legend(loc='upper right')
         ax_histogram.legend(loc='upper right')
         fig.savefig(path+'\\'+ name2fig)