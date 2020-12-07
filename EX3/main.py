# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 17:33:54 2020

@author: Yonatan Meir
"""
import pandas as pd
from pystan import StanModel
import numpy as np
from plot_chains_from_df import plot_mcmc
import os
from hpd import hpd_grid
from copy import deepcopy
path=os.getcwd()
#%%  6.1 a :
df=pd.read_csv('ShohatOphirKAMH2012dataReduced.csv')
df_mated=df.loc[df['Group']=='MatedGrouped']
df_rejected=df.loc[df['Group']!='MatedGrouped']
# model for rejected flies: 
model_rej= """
data {
    int<lower=1> N;
    real y[N];
    real mu_prior_mu;
    real mu_prior_sigma;
    real  nu_prior_rate;
    real  sigma_prior_lb;
    real  sigma_prior_rb;
    
}
parameters {
    real <lower = 2> nu;
    real <lower = 0,upper = 2> sigma;
    real  mu;
}
model {
    mu ~ normal(mu_prior_mu,mu_prior_sigma);
    sigma ~ uniform(sigma_prior_lb, sigma_prior_rb);
    nu ~ exponential(nu_prior_rate);
    y ~ student_t(nu, mu, sigma);

}
"""
# set a dict
data_rej = {'N': df_rejected.shape[0] , 'y': df_rejected['PreferenceIndex'],'mu_prior_mu' :0 ,'mu_prior_sigma':0.5, 'nu_prior_rate' : 1/30, 'sigma_prior_lb' : 0, 'sigma_prior_rb' : 2}

# Compile the model
compiled_model_rej = StanModel(model_code=model_rej)

# Fit the model (fitting 4 model with 1 chain, because there is a bug if iset chains=4)
fit_rej_1 = compiled_model_rej.sampling(data=data_rej,chains=1, iter=2000, warmup=200)
fit_rej_2 = compiled_model_rej.sampling(data=data_rej,chains=1, iter=2000, warmup=200)
fit_rej_3 = compiled_model_rej.sampling(data=data_rej,chains=1, iter=2000, warmup=200)
fit_rej_4 = compiled_model_rej.sampling(data=data_rej,chains=1, iter=2000, warmup=200)
fit_rej_dict={'1':fit_rej_1.to_dataframe(),'2':fit_rej_2.to_dataframe(),'3':fit_rej_3.to_dataframe(),'4':fit_rej_4.to_dataframe()}
# saving figures with the results using my own function
ess_dict={'nu':np.array([fit_rej_1.summary()['summary'][0,8],fit_rej_2.summary()['summary'][0,8],fit_rej_3.summary()['summary'][0,8],fit_rej_4.summary()['summary'][0,8]]),'sigma':np.array([fit_rej_1.summary()['summary'][1,8],fit_rej_2.summary()['summary'][1,8],fit_rej_3.summary()['summary'][1,8],fit_rej_4.summary()['summary'][1,8]]),'mu':np.array([fit_rej_1.summary()['summary'][2,8],fit_rej_2.summary()['summary'][2,8],fit_rej_3.summary()['summary'][2,8],fit_rej_4.summary()['summary'][2,8]])}
plot_mcmc(fit_rej_dict,'nu',path,name2fig='nu_rej',ess=ess_dict)
plot_mcmc(fit_rej_dict,'mu',path,name2fig='mu_rej',ess=ess_dict)
plot_mcmc(fit_rej_dict,'sigma',path,name2fig='sigma_rej',ess=ess_dict)

#repeating the same code for the mating flies:
model_mate= """
data {
    int<lower=1> N;
    real y[N];
    real mu_prior_mu;
    real mu_prior_sigma;
    real  nu_prior_rate;
    real  sigma_prior_lb;
    real  sigma_prior_rb;
    
}
parameters {
    real <lower = 2> nu;
    real <lower = 0,upper = 2> sigma;
    real  mu;
}
model {
    mu ~ normal(mu_prior_mu,mu_prior_sigma);
    sigma ~ uniform(sigma_prior_lb, sigma_prior_rb);
    nu ~ exponential(nu_prior_rate);
    y ~ student_t(nu, mu, sigma);

}
"""
# set a dict
data_mate = {'N': df_mated.shape[0] , 'y': df_mated['PreferenceIndex'],'mu_prior_mu' :0 ,'mu_prior_sigma':0.5, 'nu_prior_rate' : 1/30, 'sigma_prior_lb' : 0, 'sigma_prior_rb' : 2}

# Compile the model
compiled_model_rej = StanModel(model_code=model_mate)

# Fit the model (fitting 4 model with 1 chain, because there is a bug if iset chains=4)
fit_mate_1 = compiled_model_rej.sampling(data=data_mate,chains=1, iter=2000, warmup=200)
fit_mate_2 = compiled_model_rej.sampling(data=data_mate,chains=1, iter=2000, warmup=200)
fit_mate_3 = compiled_model_rej.sampling(data=data_mate,chains=1, iter=2000, warmup=200)
fit_mate_4 = compiled_model_rej.sampling(data=data_mate,chains=1, iter=2000, warmup=200)
fit_mate_dict={'1':fit_mate_1.to_dataframe(),'2':fit_mate_2.to_dataframe(),'3':fit_mate_3.to_dataframe(),'4':fit_mate_4.to_dataframe()}
# saving figures with the results using my own function
ess_dict={'nu':np.array([fit_mate_1.summary()['summary'][0,8],fit_mate_2.summary()['summary'][0,8],fit_mate_3.summary()['summary'][0,8],fit_mate_4.summary()['summary'][0,8]]),'sigma':np.array([fit_mate_1.summary()['summary'][1,8],fit_mate_2.summary()['summary'][1,8],fit_rej_3.summary()['summary'][1,8],fit_mate_4.summary()['summary'][1,8]]),'mu':np.array([fit_mate_1.summary()['summary'][2,8],fit_mate_2.summary()['summary'][2,8],fit_rej_3.summary()['summary'][2,8],fit_rej_4.summary()['summary'][2,8]])}
plot_mcmc(fit_mate_dict,'nu',path,name2fig='nu_mate',ess=ess_dict)
plot_mcmc(fit_mate_dict,'mu',path,name2fig='mu_mate',ess=ess_dict)
plot_mcmc(fit_mate_dict,'sigma',path,name2fig='sigma_mate',ess=ess_dict)

# analyzing the mean and scale diff: 
diff_df_1=deepcopy(fit_mate_1.to_dataframe())
diff_df_2=deepcopy(fit_mate_2.to_dataframe())
diff_df_3=deepcopy(fit_mate_3.to_dataframe())
diff_df_4=deepcopy(fit_mate_4.to_dataframe())
diff_df_1.loc[:,'mu_diff']=fit_rej_1.to_dataframe()['mu']-fit_mate_1.to_dataframe()['mu']
diff_df_1.loc[:,'sigma_diff']=fit_rej_1.to_dataframe()['sigma']-fit_mate_1.to_dataframe()['sigma']
diff_df_2.loc[:,'mu_diff']=fit_rej_2.to_dataframe()['mu']-fit_mate_2.to_dataframe()['mu']
diff_df_2.loc[:,'sigma_diff']=fit_rej_2.to_dataframe()['sigma']-fit_mate_2.to_dataframe()['sigma']
diff_df_3.loc[:,'mu_diff']=fit_rej_3.to_dataframe()['mu']-fit_mate_3.to_dataframe()['mu']
diff_df_3.loc[:,'sigma_diff']=fit_rej_3.to_dataframe()['sigma']-fit_mate_3.to_dataframe()['sigma']
diff_df_4.loc[:,'mu_diff']=fit_rej_4.to_dataframe()['mu']-fit_mate_4.to_dataframe()['mu']
diff_df_4.loc[:,'sigma_diff']=fit_rej_4.to_dataframe()['sigma']-fit_mate_4.to_dataframe()['sigma']
diff_dict={'1':diff_df_1,'2':diff_df_2,'3':diff_df_3,'4':diff_df_4}
plot_mcmc(diff_dict,'mu_diff',path,name2fig='mu_diff',rope=np.array([-0.1,0.1]))
