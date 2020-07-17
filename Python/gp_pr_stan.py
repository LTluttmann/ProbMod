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
      for (n in 1:N)
        Sigma[n, n] = Sigma[n,n] + sigma;
      L_Sigma = cholesky_decompose(Sigma);
      K_div_y_is = mdivide_left_tri_low(L_Sigma, y_is);
      K_div_y_is = mdivide_right_tri_low(K_div_y_is',L_Sigma)';
      k_x_is_x_pred = cov_exp_quad(x_is, x_pred, alpha, length_scale);
      f_pred_mu = (k_x_is_x_pred' * K_div_y_is); 
      v_pred = mdivide_left_tri_low(L_Sigma, k_x_is_x_pred);
      cov_f_pred = cov_exp_quad(x_pred, alpha, length_scale) - v_pred' * v_pred;
      nug_pred = diag_matrix(rep_vector(1e-10, N_pred));

      f_pred = multi_normal_rng(f_pred_mu, cov_f_pred + nug_pred);
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
    t[i] = log(y[i] + 1) - 1 / (y[i] + 1);
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
  vector[N] y_reg;
  length_scale ~ gamma(2, 20);
  alpha ~ normal(0, 1);
  f_eta ~ normal(0, 1);
  sigma ~ normal(0,1);
  y ~ poisson_log(f);
}
generated quantities {
  vector[N_pred] f_pred;
  int y_pred[N_pred];
  f_pred = gp_pred_rng(x_pred, t, x, alpha, length_scale, sigma);
  for (n in 1:N_pred)
     y_pred[n] = poisson_log_rng(f_pred[n]);
}
"""
