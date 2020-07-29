"""
Code to implement the stan model for a gaussian process poisson regression as proposed
by Chan & Vasconcelos (2009), available under: http://visal.cs.cityu.edu.hk/static/pubs/conf/iccv09-bpr.pdf
Parts of the code are related to code from Michael Betancourt, that is publicly available
under https://betanalpha.github.io/assets/case_studies/gp_part1/part1.html.

The Stan code to generate predictions on new observations of x* . The function
gen_pred_gp implements the algorithm proposed by C. E. Rasmussen &
C. K. I. Williams in Gaussian Processes for Machine Learning, p. 19
(http://www.gaussianprocess.org/gpml/chapters/RW2.pdf). Following Chan & Vasconcelos (2009)
(http://visal.cs.cityu.edu.hk/static/pubs/conf/iccv09-bpr.pdf), we take the logarithm
of y as an input to the function gen_pred_gp.
"""


gp_mod = """
functions {
  vector gp_pred_rng(row_vector[] x_pred,
                     vector y_is,
                     row_vector[] x_is,
                     real alpha,
                     real length_scale,
                     real sigma) {
    vector[size(x_pred)] f_pred;
    int N_pred;
    int N;
    N_pred = size(x_pred);
    N = rows(y_is);

    {
      matrix[N, N] L_Sigma;
      vector[N] K_div_y_is;
      matrix[N, N_pred] k_x_is_x_pred;
      matrix[N, N_pred] v_pred;
      vector[N_pred] f_pred_mu;
      matrix[N_pred, N_pred] cov_f_pred;
      matrix[N_pred, N_pred] nug_pred;
      matrix[N, N] Sigma;
      Sigma = cov_exp_quad(x_is, alpha, length_scale);
      Sigma = Sigma + diag_matrix(rep_vector(sigma, N));
      L_Sigma = cholesky_decompose(Sigma);
      K_div_y_is = mdivide_left_tri_low(L_Sigma, y_is);
      K_div_y_is = mdivide_right_tri_low(K_div_y_is',L_Sigma)';
      k_x_is_x_pred = cov_exp_quad(x_is, x_pred, alpha, length_scale);
      f_pred_mu = (k_x_is_x_pred' * K_div_y_is); 
      v_pred = mdivide_left_tri_low(L_Sigma, k_x_is_x_pred);
      cov_f_pred = cov_exp_quad(x_pred, alpha, length_scale) - v_pred' * v_pred;
      nug_pred = diag_matrix(rep_vector(1e-12, N_pred));

      f_pred = multi_normal_rng(f_pred_mu, cov_f_pred + nug_pred);
    }
    return f_pred;
  }
}
data {
    int<lower=1> N;
    int<lower=1> D;
    int<lower=1> N_pred;
    vector[N] y;
    row_vector[D] x[N];
    vector[N] zeros;
    row_vector[D] x_pred[N_pred];
}
parameters {
    real<lower=0> length_scale;
    real<lower=0> alpha;
    vector[N] f_eta;
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] f;
    {
        matrix[N, N] L_cov;
        matrix[N, N] cov;
        cov = cov_exp_quad(x, alpha, length_scale);
        for (n in 1:N)
            cov[n, n] = cov[n, n] + 1e-12;
        L_cov = cholesky_decompose(cov);
        f = L_cov * f_eta;
    }
}
model {
    length_scale ~ gamma(5, 5);
    alpha ~ normal(0, 1);
    f_eta ~ normal(0, 1);
    sigma ~ normal(0, 1);
    y ~ normal(f, sigma);
}
generated quantities {
    vector[N_pred] f_pred;
    vector[N_pred] y_pred;
    f_pred = gp_pred_rng(x_pred, y, x, alpha, length_scale, sigma);
    for (n in 1:N_pred)
        y_pred[n] = normal_rng(f_pred[n], sigma);
}
"""


hierarchical_gp_mod = """
functions {
  vector gp_pred_rng(row_vector[] x_pred,
                     vector y_is,
                     row_vector[] x_is,
                     real alpha,
                     real length_scale,
                     real sigma) {
    vector[size(x_pred)] f_pred;
    int N_pred;
    int N;
    N_pred = size(x_pred);
    N = rows(y_is);

    {
      matrix[N, N] L_Sigma;
      vector[N] K_div_y_is;
      matrix[N, N_pred] k_x_is_x_pred;
      matrix[N, N_pred] v_pred;
      vector[N_pred] f_pred_mu;
      matrix[N_pred, N_pred] cov_f_pred;
      matrix[N_pred, N_pred] nug_pred;
      matrix[N, N] Sigma;
      Sigma = cov_exp_quad(x_is, alpha, length_scale);
      Sigma = Sigma + diag_matrix(rep_vector(sigma, N));
      L_Sigma = cholesky_decompose(Sigma);
      K_div_y_is = mdivide_left_tri_low(L_Sigma, y_is);
      K_div_y_is = mdivide_right_tri_low(K_div_y_is',L_Sigma)';
      k_x_is_x_pred = cov_exp_quad(x_is, x_pred, alpha, length_scale);
      f_pred_mu = (k_x_is_x_pred' * K_div_y_is); 
      v_pred = mdivide_left_tri_low(L_Sigma, k_x_is_x_pred);
      cov_f_pred = cov_exp_quad(x_pred, alpha, length_scale) - v_pred' * v_pred;
      nug_pred = diag_matrix(rep_vector(1e-12, N_pred));

      f_pred = multi_normal_rng(f_pred_mu, cov_f_pred + nug_pred);
    }
    return f_pred;
  }
}
data {
    int<lower=1> N;
    int<lower=1> D;
    int<lower=1> C;
    int<lower=1> N_pred;
    vector[N] y;
    row_vector[D] x[N];
    int<lower=1,upper=C> c_ind[N]; // country index on training set
    vector[N] zeros;
    row_vector[D] x_pred[N_pred];
    int<lower=1,upper=C> c_ind_test[N_pred]; // country index on test set
}
parameters {
    real<lower=0> length_scale;
    real<lower=0> alpha;
    vector[N] f_eta;
    real<lower=0> sigma;
    real a[C];
}
transformed parameters {
    vector[N] f;
    {
    matrix[N, N] L_cov;
    matrix[N, N] cov;
    cov = cov_exp_quad(x, alpha, length_scale);
    for (n in 1:N)
        cov[n, n] = cov[n, n] + 1e-12;
    L_cov = cholesky_decompose(cov);
    f = L_cov * f_eta;
    }
}
model {
    length_scale ~ inv_gamma(5, 5);
    alpha ~ normal(0, 1);
    f_eta ~ normal(0, 1);
    a ~ normal(0, 1);
    sigma ~ normal(0,1);
    for (i in 1:N)
        y[i] ~ normal(a[c_ind[i]] + f[i], sigma);
}
generated quantities {
    vector[N_pred] f_pred;
    vector[N_pred] y_pred;
    f_pred = gp_pred_rng(x_pred, y, x, alpha, length_scale, sigma);
    for (i in 1:N_pred)
        y_pred[i] = normal_rng(a[c_ind_test[i]] + f_pred[i], sigma);
}
"""
