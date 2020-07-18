"""
This STAN model implements the baseline model, presented by Neelemegham and Chintagunta (1999)
"""

hierarchical_model = '''
data {
    int<lower=1> D;
    int<lower=1> C;
    int<lower=0> N;
    int<lower=0> y[N];
    int<lower=0,upper=C-1> c[N];
    row_vector[D] x[N];
    int<lower=0> N_test;
    row_vector[D] x_test[N_test];
    int<lower=0,upper=C-1> c_test[N_test];
}
parameters {
    real mu;
    real<lower=0> sigma;
    real<lower=0> a;
    real<lower=0> b;
    real mu_beta[D];
    real<lower=0> sigma_beta[D];
    real<lower=0> sigma_alpha;
    vector[D] beta[C];
    real alpha[C];
}

transformed parameters {
  real lp[N];
  real lambda[N];
  for (i in 1:N) 
    lp[i] = alpha[c[i]+1] + x[i] * beta[c[i]+1] ;
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

    y ~ poisson(lambda);
}
generated quantities {
  vector<lower=0>[N_test] y_pred;
  real lp_test[N_test];
  real lambda_test[N_test];
  for (i in 1:N_test)
    lp_test[i] = alpha[c_test[i]+1] + x_test[i] * beta[c_test[i]+1] ;
  for (i in 1:N_test) 
    lambda_test[i] = exp(lp_test[i]);
  for (i in 1:N_test) {
    y_pred[i] = poisson_rng(lambda_test[i]);
  }
}
'''