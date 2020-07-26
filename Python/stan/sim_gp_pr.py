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

sim_data_hiera_gp_pr = """
data {
    int<lower=1> N;
    int<lower=1> C;
    int<lower=1,upper=C> cc[N];
    int<lower=1> D;
    vector[D] x[N];
    // real<lower=0> a[C];
}
transformed data {
    vector[N] zeros;
    zeros = rep_vector(0, N);
}
model {}
generated quantities {
    vector[N] f;
    int y_sim[N];
    real<lower=0> alpha;
    real<lower=0> sigma;
    real<lower=0> length_scale;
    real<lower=0> a[C];
    for (c in 1:C)
        a[c] = weibull_rng(2,1);
    alpha = weibull_rng(2,1);
    sigma = weibull_rng(2,1);
    length_scale = inv_gamma_rng(5, 5);
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
        f[n] = a[cc[n]] + f[n];
    for (n in 1:N)
        y_sim[n] = poisson_log_rng(f[n]);
}
"""

sim_data_hiera_gp_norm = """
data {
    int<lower=1> N;
    int<lower=1> C;
    int<lower=1,upper=C> cc[N];
    int<lower=1> D;
    vector[D] x[N];
}
transformed data {
    vector[N] zeros;
    zeros = rep_vector(0, N);
}
model {}
generated quantities {
    vector[N] f;
    real y_sim[N];
    real<lower=0> alpha;
    real<lower=0> sigma;
    real<lower=0> length_scale;
    real a[C];
    real<lower=0> sigma_out = inv_gamma_rng(4,4);
    for (c in 1:C)
        a[c] = normal_rng(0,1);
    alpha = weibull_rng(2,1);
    sigma = weibull_rng(2,1);
    length_scale = inv_gamma_rng(5, 5);
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
        f[n] = a[cc[n]] + f[n];
    for (n in 1:N)
        y_sim[n] = normal_rng(f[n], sigma_out);
}
"""