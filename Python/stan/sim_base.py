sim_data_base_mod = """
data {
    int<lower=1> N; 
    int<lower=1> C;
    int<lower=1,upper=C> cc[N];
    int<lower=1> D;
    row_vector[D] x[N];
}
parameters {}
model {}
generated quantities {
    vector[D] beta[C];
    real y_sim[N];
    real alpha[C];
    real sigma_alpha;
    real mu_beta[D];
    real<lower=0> sigma_beta[D];
    
    sigma_alpha = inv_gamma_rng(4, 4);
    for (j in 1:D)
        sigma_beta[j] = inv_gamma_rng(4, 4);
    for (j in 1:D)
        mu_beta[j] = normal_rng(0, 10);
    for (j in 1:D)
        for(c in 1:C)
            beta[c, j] = normal_rng(mu_beta[j], sigma_beta[j]);
    for(c in 1:C)
        alpha[c] = normal_rng(0, sigma_alpha);
    for (i in 1:N)
        y_sim[i] = poisson_log_rng(alpha[cc[i]] + x[i] * beta[cc[i]]);
}
"""

sim_data_horseshoe = """
data {
    int<lower=1> N; 
    int<lower=1> C;
    int<lower=1,upper=C> cc[N];
    int<lower=1> D;
    real<lower=0> sigma_squared;
    row_vector[D] x[N];
}
parameters {}
model {}
generated quantities {
    real<lower=0> sigma_beta[D];
    vector[D] beta[C];
    real y[N];

    real omega;
    real lambda_squared;
    real gamma_param;
    real alpha[C];
    real sigma_alpha;

    gamma_param = gamma_rng(0.5, sigma_squared);
    lambda_squared = inv_gamma_rng(0.5, gamma_param);
    omega = gamma_rng(0.5, lambda_squared);
    sigma_alpha = inv_gamma_rng(0.5, gamma_param);

    for (j in 1:D)
        sigma_beta[j] = inv_gamma_rng(0.5, omega);

    for (j in 1:D)
        for(c in 1:C)
            beta[c, j] = normal_rng(0, sigma_beta[j]);
    for(c in 1:C)
        alpha[c] = normal_rng(0, sigma_alpha);
    for (i in 1:N)
        y[i] = poisson_log_rng(alpha[cc[i]] + x[i] * beta[cc[i]]);
}
"""