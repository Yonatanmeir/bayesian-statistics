# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:03:51 2020

@author: Yonatan Meir
"""
import numpy as np
import math
from metropolis import metropolis_fcn,gibbs_sampler
import matplotlib.pyplot as plt
from scipy.stats import norm
from hpd import hpd_grid
def binomial_posterior(theta):
    try:
        if theta.shape[0]>1:
                likelihood=np.multiply(np.multiply(theta,(1-theta)),(1-theta))
                prior=1/1.5 * (np.multiply(np.cos(4*math.pi*theta)+1,np.cos(4*math.pi*theta)+1))
                return prior, np.multiply(prior,likelihood)
        else:
            if theta>1 or theta<0:
                return 0
            else:
                likelihood=theta*(1-theta)**2
                prior=1/1.5 * (np.cos(4*math.pi*theta)+1)**2
                return likelihood*prior
    except:
        if theta>1 or theta<0:
            return 0
        else:
            likelihood=theta*(1-theta)**2
            prior=1/1.5 * (np.cos(4*math.pi*theta)+1)**2
            return likelihood*prior
def normal_posterior(param):
    mu=param[0]
    sigma=param[1]
    mu_0=0
    sigma_0=np.sqrt(100)
    nu_0=1
    s_0=1
    np.random.seed(1)
    data=np.random.normal(loc=0,scale=1,size=(20))
    np.random.seed()
    likelihood=np.cumprod(norm.pdf(data,loc=mu,scale=sigma))[-1]
    prior_mu=norm.pdf(mu,loc=mu_0,scale=sigma_0)
    #tmp=np.random.chisquare(nu_0,size=(1))
    prior_sigma=((sigma**2)**(-nu_0/2 -1))*np.exp(-nu_0*s_0**2/(2*sigma**2))
    return likelihood*prior_mu*prior_sigma
def conditional_mu(mu,sigma):
    np.random.seed(1)
    data=np.random.normal(loc=0,scale=1,size=(20))
    np.random.seed()
    mu_0=0
    sigma_0=np.sqrt(100)
    nu_0=1
    s_0=1
    tau_0=1/sigma_0**2
    tau=1/sigma**2
    y_mean=np.mean(data)
    mu_n=(tau_0*mu_0+data.shape[0]*tau*y_mean)/(tau_0+data.shape[0]*tau)
    sigma_n=1/np.sqrt(tau_0+data.shape[0]*tau)
    return np.random.normal(loc=mu_n,scale=sigma_n,size=(1))

def conditional_sigma(mu,sigma):
    np.random.seed(1)
    data=np.random.normal(loc=0,scale=1,size=(20))
    np.random.seed()
    mu_0=0
    sigma_0=np.sqrt(100)
    nu_0=1
    s_0=1
    var_y=np.mean(np.multiply(data-mu,data-mu))
    nu_n=nu_0+data.shape[0]
    s_n=(nu_0*s_0**2+data.shape[0]*var_y)/nu_n
    return (s_n*nu_n)/np.random.chisquare(nu_n)
#main : 

        
        
#2: 
fig_1=plt.figure(figsize=(10,10))
axis=fig_1.add_subplot(111)
pr,post=binomial_posterior(np.linspace(0,1,10000))
axis.plot(np.linspace(0,1,10000),pr,color='r',label='prior')
axis.plot(np.linspace(0,1,10000),post,color='b',label='posterior')
axis.set_xlabel('theta')
axis.set_ylabel('unormalized density')
fig_1.savefig('ex2')



#3:
prop_width=np.array([0.02,0.2])
init_pos=np.array([0.01,0.5])
fig=plt.figure(figsize=(10,10))
fig_ind=1
for wid in prop_width:
    for init in init_pos:
        cur_ax=fig.add_subplot(4,2,fig_ind)
        fig_ind+=1
        cur_ax2=fig.add_subplot(4,2,fig_ind)
        fig_ind+=1
        res=metropolis_fcn(binomial_posterior,init,wid,10000)
        cur_ax.plot(res,np.arange(res.shape[0]),label='width='+str(wid)+' ,initial pos='+str(init))
        cur_ax.set_xlabel('theta')
        cur_ax.set_ylabel('step')
        cur_ax.set_title('mcmc')
        
        cur_ax2.hist(res,bins=100,label='width='+str(wid)+' ,initial pos='+str(init))
        cur_ax2.set_title('histogram')
        cur_ax.legend(loc='upper left')
        cur_ax2.legend(loc='upper left')
fig.savefig('ex3')

# 5:
fig_4=plt.figure(figsize=(10,10))
cur_ax=fig_4.add_subplot(2,1,1)
cur_ax2=fig_4.add_subplot(2,1,2)
res=metropolis_fcn(normal_posterior,np.array([0,1]),0.02,5000)
a=hpd_grid(res[:,0])
mu_l=a[0][0][0]
mu_r=a[0][0][1]
b=hpd_grid(res[:,1])
sigma_l=b[0][0][0]
sigma_r=b[0][0][1]

cur_ax.hist(res[:,0],bins=50,label='mu')
cur_ax.plot(np.array([mu_l,mu_l]),np.array([0,400]),color='r',label='hdi=['+str(mu_l)+'  '+str(mu_r)+']')
cur_ax.plot(np.array([mu_r,mu_r]),np.array([0,400]),color='r')
cur_ax.set_title('histogram of mu with metropolis algorithm')
cur_ax.set_xlabel('mu')
cur_ax.set_ylabel('p(mu|data)')
cur_ax.legend()

cur_ax2.hist(res[:,1],bins=50,label='sigma')
cur_ax2.plot(np.array([sigma_l,sigma_l]),np.array([0,400]),color='r',label='hdi=['+str(sigma_l)+'  '+str(sigma_r)+']')
cur_ax2.plot(np.array([sigma_r,sigma_r]),np.array([0,400]),color='r')
cur_ax2.set_title('histogram of sigma with metropolis algorithm')
cur_ax2.legend(loc='upper right')
cur_ax2.set_xlabel('sigma')
cur_ax2.set_ylabel('p(sigma|data)')
fig_4.savefig('ex5')

# 6:
fig_6=plt.figure(figsize=(10,10))
cur_ax=fig_6.add_subplot(2,1,1)
cur_ax2=fig_6.add_subplot(2,1,2)
res1,res2=gibbs_sampler(np.array([0,1]),5000,conditional_mu,conditional_sigma)
a=hpd_grid(res1)
mu_l=a[0][0][0]
mu_r=a[0][0][1]
b=hpd_grid(res2)
sigma_l=b[0][0][0]
sigma_r=b[0][0][1]

cur_ax.hist(res1,bins=50,label='mu')
cur_ax.plot(np.array([mu_l,mu_l]),np.array([0,3000]),color='r',label='hdi=['+str(mu_l)+'  '+str(mu_r)+']')
cur_ax.plot(np.array([mu_r,mu_r]),np.array([0,3000]),color='r')
cur_ax.set_title('histogram of mu with gibbs sampler')
cur_ax.set_xlabel('mu')
cur_ax.set_ylabel('p(mu|data)')

cur_ax.legend()

cur_ax2.hist(res2,bins=50,label='sigma')
cur_ax2.plot(np.array([sigma_l,sigma_l]),np.array([0,3000]),color='r',label='hdi=['+str(sigma_l)+'  '+str(sigma_r)+']')
cur_ax2.plot(np.array([sigma_r,sigma_r]),np.array([0,3000]),color='r')
cur_ax2.set_title('histogram of sigma with gibbs sampler')
cur_ax2.legend()
cur_ax2.set_xlabel('sigma')
cur_ax2.set_ylabel('p(sigma|data)')
fig_6.savefig('ex6')
