# -*- coding: utf-8 -*-
"""
This module simulates data and back estimates the parameters for baseline model (Neelamegham) and plots results
"""

import pystan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from helper_func_diagnostics import plot_trace, get_pred_plot_with_conf
from stan.orig_pois_reg_stan import hierarchical_model_pois
np.random.seed(178153)


def sim_df(num_movies, num_countries, num_periods):
    SCREENS = np.empty((num_movies, num_countries, num_periods))
    for m in range(num_movies):
        for c in range(num_countries):
            for t in range(num_periods):
                SCREENS[m, c, t] = max(6, np.random.normal(20, 2))

    DIST = np.empty((num_movies, num_countries, num_periods))
    for m in range(num_movies):
        for c in range(num_countries):
            DIST[m, c] = np.random.randint(0, 2)

    TREND = np.empty((num_movies, num_countries, num_periods))
    for m in range(num_movies):
        for c in range(num_countries):
            init_release = np.random.randint(0, num_periods)
            # TREND[m,c] = np.append(np.zeros(init_release), np.array(range(0, num_periods-init_release)))
            TREND[m, c] = np.random.randint(0, 2)

    m = list(range(num_movies))
    c = list(range(num_countries))
    t = list(range(num_periods))
    index = pd.MultiIndex.from_product([m, c, t], names=["m", "c", "t"])

    df = pd.DataFrame(index=index)
    df['TREND'] = TREND.ravel()
    df['SCREENS'] = SCREENS.ravel()
    df['DIST'] = DIST.ravel()
    return df


def sim(df):
    a_prior = 4
    b_prior = 1
    mu_prior = 1
    sigma_prior = 2

    num_params = len(df.columns)
    num_countries = len(df.index.get_level_values('c').unique())
    num_movies = len(df.index.get_level_values('m').unique())
    num_periods = len(df.index.get_level_values('t').unique())

    mu_beta = np.random.normal(mu_prior, sigma_prior, num_params)
    sigma_beta = 1 / np.random.gamma(a_prior, b_prior, num_params)

    beta_j_c = np.empty((num_params, num_countries))
    for i in range(num_params):  # todo loop über perioden
        mu_beta_country = np.random.normal(mu_beta[i], sigma_beta[i], num_countries)
        beta_j_c[i,] = mu_beta_country

    sigma_alpha = 1 / np.random.gamma(a_prior, b_prior, num_countries)
    alpha_c = np.random.normal(0, sigma_alpha)

    alpha_c = np.empty(num_countries)
    for i in range(num_countries):
        alpha_c[i] = np.random.normal(0, sigma_alpha[i], 1)

    log_view_intensity = np.empty((num_movies, num_countries, num_periods))
    for m in range(num_movies):
        for c in range(num_countries):
            for t in range(num_periods):
                log_view_intensity[m, c, t] = (alpha_c[c] +
                                               beta_j_c[0, c] * df.TREND.loc[m, c, t] +
                                               beta_j_c[1, c] * np.log(df.SCREENS.loc[m, c, t]) +
                                               beta_j_c[2, c] * df.DIST.loc[m, c, t])

    print(alpha_c)
    print(beta_j_c)
    view_intesity = np.exp(log_view_intensity)
    view_count = np.empty((num_movies, num_countries, num_periods))
    for m in range(num_movies):
        for c in range(num_countries):
            for t in range(num_periods):
                view_count[m, c, t] = np.random.poisson(view_intesity[m, c, t])

    return view_count, alpha_c, beta_j_c


num_movies = 3
num_countries = 3
num_periods = 25

df = sim_df(num_movies, num_countries, num_periods)

view_count, alpha, beta = sim(df)

a = view_count[2, 0][:]
plt.plot(a)
plt.show()

"""# Model"""

hierarchical_model = '''
data {
    int<lower=1> D;
    int<lower=1> C;
    int<lower=0> N;
    int<lower=0> y[N];
    int<lower=1,upper=C> c[N];
    row_vector[D] x[N];
    int<lower=0> N_test;
    row_vector[D] x_test[N_test];
    int<lower=1,upper=C> c_test[N_test];
}
parameters {
    real mu;
    real sigma;
    real a;
    real b;
    real mu_beta[D];
    real sigma_beta[D];
    real sigma_alpha;
    vector[D] beta[C];
    real alpha[C];
}

transformed parameters {
  real lp[N];
  real lambda[N];

  for (i in 1:N) 
    lp[i] = alpha[c[i]] + x[i] * beta[c[i]] ;
  for (i in 1:N) 
    lambda[i] = exp(lp[i]);
}

model {
    mu_beta ~ normal(mu, sigma);
    sigma_beta ~ inv_gamma(a, b);
    sigma_alpha ~ inv_gamma(a, b);

    for (ci in 1:C) 
      beta[ci] ~ normal(mu_beta, sigma_beta);
    
    for (ci in 1:C) 
      alpha[ci] ~ normal(0, sigma_alpha);

    y ~ poisson_log(lp);
}
generated quantities {
  vector<lower=0>[N_test] y_pred;
  real lp_test[N_test];
  real lambda_test[N_test];
  for (i in 1:N_test)
    lp_test[i] = alpha[c_test[i]] + x_test[i] * beta[c_test[i]] ;
  for (i in 1:N_test) 
    lambda_test[i] = exp(lp_test[i]);
  for (i in 1:N_test) {
    y_pred[i] = poisson_log_rng(lp_test[i]);
  }
}
'''

DATA_PATH = "../stan_dumps/"
try:
    sm = pickle.load(open(DATA_PATH + 'pred_base.pkl', 'rb'))
except:
    sm = pystan.StanModel(model_code=hierarchical_model_pois, model_name='HLR')
    pickle.dump(sm, open(DATA_PATH + 'pred_base.pkl', 'wb'))
# sm = pystan.StanModel(model_code=hierarchical_model, model_name='HLR')
df_log = df.copy()
df_log.SCREENS = np.log(df_log.SCREENS)

df_log["target"] = view_count.ravel()

features = ["TREND", "SCREENS", "DIST"]

sample = np.random.choice(df_log.index.get_level_values('m').unique(), int(np.floor(num_movies * 0.9)), replace=False)

df_train = df_log[df_log.index.get_level_values('m').isin(sample)]

df_test = df_log[~df_log.index.get_level_values('m').isin(sample)]

set(df_train.index.get_level_values('m'))

set(df_test.index.get_level_values('m'))

data = dict(x=df_train.loc[:, features].values,
            N=df_train.shape[0],
            C=num_countries,
            D=len(features),
            c=df_train.index.get_level_values('c').values + 1,
            y=df_train.target.astype('int').ravel(),
            N_test=df_test.shape[0],
            x_test=df_test.loc[:, features].values,
            c_test=df_test.index.get_level_values('c').values+1)

try_count = 0
while try_count < 3:
    try:
        fit = sm.sampling(data=data, chains=1, iter=2000, warmup=500, n_jobs=2)
        break
    except:
        try_count += 1

summary_dict = fit.summary()
sum_df = pd.DataFrame(summary_dict['summary'],
                      columns=summary_dict['summary_colnames'],
                      index=summary_dict['summary_rownames'])

preds = []
for i in range(df_test.shape[0]):
    preds.append(sum_df.loc['y_pred[{}]'.format(i + 1)]['mean'])

pred_df = pd.DataFrame(np.vstack((df_test.target.ravel(), preds))).T

pred_df['abs_error'] = np.abs(pred_df.iloc[:, 0] - pred_df.iloc[:, 1])

pred_df['ape'] = pred_df.abs_error / pred_df.iloc[:, 0].replace(0, 1)

print(np.mean(pred_df['ape']))

pred_df.index = df_test.index
pred_df.head(10)

"""# Further Diagnostics"""
alpha_fit = fit['alpha[2]']
alpha_sim = alpha[1]
beta_fit = fit['beta[1,1]']
beta_sim = beta[0][0]
sigma = fit['sigma']
lp = fit['lp__']

plot_trace(alpha_fit, 'alpha[2]', alpha_sim)
plt.show()
plot_trace(beta_fit, 'beta', beta_sim)
plt.show()

test_samps = list(set(df.index.get_level_values('m')).difference(set(sample)))
get_pred_plot_with_conf(fit, sum_df, test_samps[-1], 0, 1000, pred_df)
plt.show()
# x = az.plot_trace(fit, var_names=["alpha","beta"])

divs = []
for i in range(len(alpha)):
    divs.append(np.abs(alpha[i] - sum_df.loc["alpha[{}]".format(i + 1)]["50%"]) / np.abs(alpha[i]))
for i in range(len(beta)):
    for j in range(len(beta[i])):
        divs.append(beta[i][j] - sum_df.loc["beta[{},{}]".format(j + 1, i + 1)]["50%"])

mean_param_div = np.mean(divs)
print("mean_param_div: ", mean_param_div)
