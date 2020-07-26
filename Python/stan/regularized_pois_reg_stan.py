"""
This STAN model implements the baseline model, presented by Neelamegham and Chintagunta (1999) and extends
it with shrinkage priors on the regression weights. Therefore, the horseshoe prior (Carvalho 2010) is used.
"""
hierarchical_model_regularized = '''
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
    real<lower=0> a; // shape parameter for diffuse prior on variance of intercept 
    real<lower=0> b; // scale parameter for diffuse prior on variance of intercept 
    real<lower=0> lambda[D];
    real<lower=0> tau_tilde;
    real<lower=0> sigma_alpha; // prior variance for intercept
    vector[D] beta[C];
    real alpha[C];
}

transformed parameters {
    real lp[N];
    for (i in 1:N) 
        lp[i] = alpha[c[i]] + x[i] * beta[c[i]] ;
}

model {
    lambda ~ cauchy(0, 1);
    tau_tilde ~ cauchy(0, 1);
    sigma_alpha ~ inv_gamma(a,b);
    for (ci in 1:C) 
        for (di in 1:D)
            beta[ci][di] ~ normal(0, lambda[di] * tau_tilde);
    for (ci in 1:C) 
        alpha[ci] ~ normal(0, sigma_alpha);
    y ~ poisson_log(lp);
}
generated quantities {
    vector<lower=0>[N_test] y_pred;
    real lp_test[N_test];
    for (i in 1:N_test)
        lp_test[i] = alpha[c_test[i]] + x_test[i] * beta[c_test[i]] ;
    for (i in 1:N_test) {
        if (lp_test[i] > 20)
            y_pred[i] = normal_rng(exp(lp_test[i]), sqrt(exp(lp_test[i])));
        else
            y_pred[i] = poisson_log_rng(lp_test[i]);
  }
}
'''