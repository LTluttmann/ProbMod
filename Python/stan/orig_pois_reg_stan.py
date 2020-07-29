"""
This STAN model implements the baseline model, presented by Neelemegham and Chintagunta (1999)
"""

hierarchical_model_pois = '''
data {
    int<lower=1> D;
    int<lower=1> C;
    int<lower=0> N;
    int<lower=0> y[N];
    int<lower=1,upper=C> c[N];
    row_vector[D] x[N];
    int<lower=0> N_test;
    row_vector[D] x_test[N_test];
    int<lower=1, upper=C> c_test[N_test];
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
    for (i in 1:N) 
        lp[i] = alpha[c[i]] + x[i] * beta[c[i]];
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
    int<lower=0> y_pred[N_test];
    real lp_test[N_test];
    for (i in 1:N_test)
        lp_test[i] = alpha[c_test[i]] + x_test[i] * beta[c_test[i]] ;
    for (i in 1:N_test)
        if (lp_test[i] > 20) // avoid error messages due to lamba being too high
            y_pred[i] = poisson_log_rng(20);
        else
            y_pred[i] = poisson_log_rng(lp_test[i]);
}
'''

hierarchical_model_normal = '''
data {
    int<lower=1> D;
    int<lower=1> C;
    int<lower=0> N;
    vector[N] y;
    int<lower=1,upper=C> c[N];
    row_vector[D] x[N];
    int<lower=0> N_test;
    row_vector[D] x_test[N_test];
    int<lower=1,upper=C> c_test[N_test];
}
parameters {
    real mu_beta[D];
    real<lower=0> sigma_beta[D];
    real<lower=0> sigma_alpha;
    vector[D] beta[C];
    real alpha[C];
    real<lower=0> sigma;
}

transformed parameters {
    real lp[N];
    for (i in 1:N) 
        lp[i] = alpha[c[i]] + x[i] * beta[c[i]];
}

model {
    mu_beta ~ normal(0, 100); // choose extremely flat priors as magnitude of data is extremely large
    sigma_beta ~ inv_gamma(0.001, 0.0001);
    sigma_alpha ~ inv_gamma(0.001, 0.0001);
    for (ci in 1:C) 
        beta[ci] ~ normal(mu_beta, sigma_beta);
    for (ci in 1:C) 
        alpha[ci] ~ normal(0, sigma_alpha);
    for (i in 1:N)
        y[i] ~ normal(lp[i], sigma);
}
generated quantities {
    real y_pred[N_test];
    real lp_test[N_test];
    for (i in 1:N_test)
        lp_test[i] = alpha[c_test[i]] + x_test[i] * beta[c_test[i]] ;
    for (i in 1:N_test)
        y_pred[i] = normal_rng(lp_test[i], sigma);
}
'''
