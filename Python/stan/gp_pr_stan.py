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
of y as an input to the function gen_pred_gp. To be more precise, y_is = log(y) + (c / c + y),
to account for observations that are equal to zero.
"""

gp_mod_pois = """
functions {
  vector gp_pred_rng(row_vector[] x_pred,
                     vector t,
                     row_vector[] x_is,
                     real alpha,
                     real length_scale) {
    vector[size(x_pred)] f_pred;
    int N_pred;
    int N;
    N_pred = size(x_pred);
    N = rows(t);

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
      Sigma = Sigma + diag_matrix(rep_vector(1e-10, N));
      L_Sigma = cholesky_decompose(Sigma);
      K_div_y_is = mdivide_left_tri_low(L_Sigma, t);
      K_div_y_is = mdivide_right_tri_low(K_div_y_is',L_Sigma)';
      k_x_is_x_pred = cov_exp_quad(x_is, x_pred, alpha, length_scale);
      f_pred = (k_x_is_x_pred' * K_div_y_is); 
    }
    return f_pred;
  }
}
data {
  int<lower=1> N;
  int<lower=1> D;
  int<lower=1> N_pred;
  int y[N];
  row_vector[D] x[N];
  vector[N] zeros;
  row_vector[D] x_pred[N_pred];
}
transformed data{
    vector[N] t;
    for (i in 1:N)
        t[i] = log(y[i]+1); // add constant term c to log(.) if any zero observations are in y 
}
parameters {
  real<lower=0> length_scale;
  real<lower=0> alpha;
  vector[N] f_eta;
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
    length_scale ~ gamma(2, 20);
    alpha ~ normal(0, 1);
    f_eta ~ normal(0, 1);
    y ~ poisson_log(f);
}
generated quantities {
    vector[N_pred] f_pred;
    vector[N_pred] y_pred;
    f_pred = gp_pred_rng(x_pred, t, x, alpha, length_scale);
    for (n in 1:N_pred)
        if (f_pred[n] > 20)
            y_pred[n] = normal_rng(exp(f_pred[n]), sqrt(exp(f_pred[n])));
        else
            y_pred[n] = poisson_log_rng(f_pred[n]);
}
"""

"""include hierarchies"""
hierarchical_gp_pr_mod = """
functions {
  vector gp_pred_rng(row_vector[] x_pred,
                     vector t,
                     row_vector[] x_is,
                     real alpha,
                     real length_scale) {
    vector[size(x_pred)] f_pred;
    int N_pred;
    int N;
    N_pred = size(x_pred);
    N = rows(t);

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
      Sigma = Sigma + diag_matrix(rep_vector(1e-10, N));
      L_Sigma = cholesky_decompose(Sigma);
      K_div_y_is = mdivide_left_tri_low(L_Sigma, t);
      K_div_y_is = mdivide_right_tri_low(K_div_y_is',L_Sigma)';
      k_x_is_x_pred = cov_exp_quad(x_is, x_pred, alpha, length_scale);
      f_pred = (k_x_is_x_pred' * K_div_y_is); 
    }
    return f_pred;
  }
}
data {
    int<lower=1> N;
    int<lower=1> D;
    int<lower=1> C;
    int<lower=1> N_pred;
    int y[N];
    row_vector[D] x[N];
    int<lower=1,upper=C> c_ind[N]; // country index on training set
    vector[N] zeros;
    row_vector[D] x_pred[N_pred];
    int<lower=1,upper=C> c_ind_test[N_pred]; // country index on test set
}
transformed data{
    vector[N] t;
    for (i in 1:N)
        t[i] = log(y[i]+1); // add constant term c to log(.) if any zero observations are in y 
}
parameters {
    real<lower=0> length_scale;
    real<lower=0> alpha;
    vector[N] f_eta;
    real<lower=0> a[C];
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
    for (i in 1:N)
        y[i] ~ poisson_log(a[c_ind[i]] + f[i]);
}
generated quantities {
    vector[N_pred] f_pred;
    vector[N_pred] y_pred;
    f_pred = gp_pred_rng(x_pred, t, x, alpha, length_scale);
    for (i in 1:N_pred)
        if (a[c_ind_test[i]] + f_pred[i] > 20)
            y_pred[i] = normal_rng(exp(a[c_ind_test[i]] + f_pred[i]), sqrt(exp(a[c_ind_test[i]] + f_pred[i])));
        else
            y_pred[i] = poisson_log_rng(a[c_ind_test[i]] + f_pred[i]);
}
"""