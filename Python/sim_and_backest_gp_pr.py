# -*- coding: utf-8 -*-
"""
Within this module, data from a poisson process using a gaussian process as prior for the mean
is simulated. The model is of the form:
f ~ Gaussian(0, K(x) + sigma^2*I)
y_i ~Poisson(exp(f_i)) for all i in {1,...,N}

We use the stan model in gp_pr_stan.py to back estimate the hyperparameters of the
Kernel K(x) as well as to to predictions on a holdout set.

Results are displayed at the end of the script, using matplotlib.
"""

# import of necessary modules
import pystan
import pandas as pd
import pickle

# import of the stan code
from gp_pr_stan import gp_mod_pois

# import own code for model diagnostics
from helper_func_diagnostics import *

sim_data = """
data {
  int<lower=1> N;
  real<lower=0> length_scale;
  real<lower=0> alpha;
  real<lower=0> sigma;
}
transformed data {
  vector[N] zeros;
  zeros = rep_vector(0, N);
}
model {}
generated quantities {
  real x[N];
  vector[N] f;
  int y_pois[N];
  for (n in 1:N)
    x[n] = uniform_rng(-2,2);
  {
    matrix[N, N] cov;
    matrix[N, N] L_cov;
    cov = cov_exp_quad(x, alpha, length_scale);
    for (n in 1:N)
    cov[n, n] = cov[n, n] + 1e-12;
    L_cov = cholesky_decompose(cov);
    f = multi_normal_cholesky_rng(zeros, L_cov);
  }
  for (n in 1:N)
    y_pois[n] = poisson_log_rng(f[n]);
}
"""

# compile stan code to simulate data from gaussian process
try:
    sm = pickle.load(open("../stan_dumps/" + 'sim_gp_pr.pkl', 'rb'))
except FileNotFoundError:
    sm = pystan.StanModel(model_code=sim_data, model_name='sim_gp_pr')
    pickle.dump(sm, open("../stan_dumps/" + 'sim_gp_pr.pkl', 'wb'))

# specify necessary data for simulation, especially the hyperparameters for the kernel
data = dict(
    N=300,
    alpha=1,
    length_scale=0.15,
    sigma=np.sqrt(0.1)
)

# simulate data from the model
draw = sm.sampling(data=data, iter=1, algorithm='Fixed_param', chains=1, seed=363360090)

# put the simulated data in a data frame
samps = draw.extract()
df = pd.DataFrame({"x": samps['x'][0],
                   "y": samps["y_pois"][0],
                   "f": np.exp(samps["f"][0])})

# sort according to x-axis for plotting purposes
df.sort_values('x', inplace=True)

# sample points
sample = np.random.choice(df.index, 60, replace=False)

# plotting of the simulated data
plt.scatter(df.loc[sample].x, df.loc[sample].y)
plt.plot(df.x, df.f, c='r')
plt.show()
# transform the observations to type integer, otherwise stan will throw errors.
df.y = df.y.astype('int')
# data for predicting new observations using x*
stan_data_pois = dict(N=len(sample), N_pred=data['N'] - len(sample),
                      zeros=np.zeros(len(sample)), x=df.loc[sample].x,
                      y=df.loc[sample].y,
                      x_pred=df.loc[~df.index.isin(sample)].x)

# compile the stan code for doing inference (gp_pr_stan.py)
try:
    gp = pickle.load(open("../stan_dumps/" + 'pred_gp_pr.pkl', 'rb'))
except FileNotFoundError:
    gp = pystan.StanModel(model_code=gp_mod_pois, model_name='pred_gp_pr')
    pickle.dump(gp, open("../stan_dumps/" + 'pred_gp_pr.pkl', 'wb'))

# do the HMC sampling (default 4 chains)
fit = gp.sampling(data=stan_data_pois, chains=1)
pois_samps = fit.extract()

# ----------------------------Do the model diagnostics-------------------------------------------
# retrieve parameters
alpha_fit = pois_samps['alpha']
sigma_fit = pois_samps['sigma']
length_scale = pois_samps['length_scale']
# plot results
plot_trace(alpha_fit, 'alpha', data['alpha'])
plt.show()
plot_trace(length_scale, 'length scale', data['length_scale'])
plt.show()
plot_trace(sigma_fit, 'sigma', data['sigma'])
plt.show()
# get model summary
summary_dict = fit.summary()
sum_df = pd.DataFrame(summary_dict['summary'],
                      columns=summary_dict['summary_colnames'],
                      index=summary_dict['summary_rownames'])
# get mean predictions per new observation
y_test = df.loc[~df.index.isin(sample)].y.values
y_hat = []
for i in range(df.loc[~df.index.isin(sample)].x.shape[0]):
    y_hat.append(sum_df.loc['y_pred[{}]'.format(i + 1)]['mean'])

get_pred_plot_with_conf(fit, 250, y_hat, y_test)
plt.show()
# plot
plt.scatter(df.loc[~df.index.isin(sample)].x, y_hat, c='r')
plt.scatter(df.loc[~df.index.isin(sample)].x, df.loc[~df.index.isin(sample)].y, c='b')
plt.show()

# calc interpretable (MAPE in [0,1])  error measure
ape = []
for i in range(len(y_hat)):
    if y_test[i] == 0:
        ape.append(np.abs(y_test[i] - y_hat[i]) / (y_test[i] + 1))
    else:
        ape.append(np.abs(y_hat[i] - y_test[i]) / y_test[i])
print("MAPE:", np.mean(ape))
