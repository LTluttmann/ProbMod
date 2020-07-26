# -*- coding: utf-8 -*-
"""
This module simulates data and back estimates the parameters
"""

# import of necessary modules
import pystan
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# import of the stan code
from stan.gp_pr_stan import gp_mod_pois, hierarchical_gp_pr_mod
from stan.gp_stan import hierarchical_gp_mod
from stan.sim_gp_pr import sim_data_hiera_gp_pr, sim_data_hiera_gp_norm

# import own code for model diagnostics
from helper_func_diagnostics import *
from train_and_predict import get_and_filter_df, get_model_data_dict

# --------------------------------- config --------------------------------------------------------

DATAMART_PATH = '../Data/final.xlsx'
SENT_PATH = '../Data/sent_scores_df.pkl'
FIGURE_PATH = '../figures/'
FEATURES = [
    'production_budget'
]
"""
, 'dist', 'stars', 'direc', 'screens',
    'opening_weekend_revenue', 'num_pos_tweets',
    'adventure', 'comedy', 'docu', 'drama', 'horror', 'musical', 'thriller',
    'action'
"""

TARGET = "worldwide_box_office"

MODEL_TO_SIM = "pred_hiera_gp"

model_code_dict = dict(
    pred_hiera_gp_pr=hierarchical_gp_pr_mod,
    pred_hiera_gp=hierarchical_gp_mod
)

sim_code_dict = dict(
    pred_hiera_gp_pr=sim_data_hiera_gp_pr,
    pred_hiera_gp=sim_data_hiera_gp_norm
)
# ----------------------------------------------------------------------------------------------------


def get_train_test(full_df, features, frac_test_set=0.2, sample_size=None, scale_factor=None):
    # some models take very long to train, sampling might be required
    if sample_size:
        full_df = full_df.loc[np.random.choice(full_df.index, sample_size)]
    # do train test splitting
    df_train, df_test = train_test_split(full_df, test_size=frac_test_set)
    # extract design matrix and target vector for both train and test set
    X_train = df_train[features]
    y_train = df_train[TARGET]
    X_test = df_test[features]
    y_test = df_test[TARGET]
    # poisson regression not appropriate for very large target observations, rescaling might be necessary
    if scale_factor:
        y_train = np.ceil(y_train / scale_factor).astype('int')
        y_test = np.ceil(y_test / scale_factor).astype('int')
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # compile stan code to simulate data from gaussian process
    try:
        sm = pickle.load(open("../stan_dumps/" + 'sim_' + MODEL_TO_SIM + '.pkl', 'rb'))
    except FileNotFoundError:
        sm = pystan.StanModel(model_code=sim_code_dict[MODEL_TO_SIM], model_name='sim_' + MODEL_TO_SIM)
        pickle.dump(sm, open("../stan_dumps/" + 'sim_' + MODEL_TO_SIM + '.pkl', 'wb'))
    df = get_and_filter_df(DATAMART_PATH, SENT_PATH)
    df = df.loc[np.random.choice(df.index, 120, replace=False)]
    scaler = StandardScaler()
    X = np.random.uniform(-2,2, 120)
    # specify necessary data for simulation
    data = dict(
        N=df.shape[0],
        C=2,
        x=X.reshape(-1, 1),
        cc=[1]*int(np.floor(X.shape[0] / 2)) + [2] * int(np.ceil(X.shape[0] / 2)),  # randomly split into 2 countries
        D=len(FEATURES)
    )
    # simulate data from the model
    draw = sm.sampling(data=data, iter=1, algorithm='Fixed_param', chains=1, seed=363360090)

    # put the simulated data in a data frame
    samps = draw.extract()
    df_sim = pd.concat([pd.DataFrame(X, columns=FEATURES),
                        pd.DataFrame({TARGET: samps["y_sim"][0],
                                      "f": samps["f"][0]})],  # np.exp() in case of poisson
                       axis=1)
    df_sim["country"] = data["cc"]

    # sort according to x-axis for plotting purposes
    df_sim_plot = df_sim[["production_budget", TARGET, "f", "country"]].sort_values('production_budget')
    df_sim_plot.country.replace({1: "USA", 2: "GB"}, inplace=True)
    palette = {"USA": sns.color_palette("cubehelix", 8)[0],
               "GB": sns.color_palette("cubehelix", 8)[6]}

    # plot simulated data
    try:
        sample = np.random.choice(df_sim.index, 60, replace=False)
    except:
        sample = df_sim.index
    _ = sns.scatterplot(data=df_sim_plot.loc[sample], x='production_budget', y=TARGET, hue='country', palette=palette)
    _ = sns.lineplot(df_sim_plot.production_budget, df_sim_plot.f, hue=df_sim_plot.country, palette=palette, legend=False)
    fig = _.get_figure()
    fig.savefig(FIGURE_PATH + "sim_" + MODEL_TO_SIM)

    # data for predicting new observations using x*
    df_sim.set_index('country', inplace=True)
    # convert target variable to type integer
    if MODEL_TO_SIM == 'pred_hiera_gp_pr':
        df_sim[TARGET] = df_sim[TARGET].astype('int')
    X_train, X_test, y_train, y_test = get_train_test(df_sim, FEATURES, TARGET)
    # get the data for fitting the model
    stan_data = get_model_data_dict(MODEL_TO_SIM, X_train, X_test, y_train, C=len(pd.unique(X_train.index)),
                                    c_ind=X_train.index, c_test_ind=X_test.index)
    # compile the stan code for doing inference (gp_pr_stan.py)
    try:
        gp = pickle.load(open("../stan_dumps/" + MODEL_TO_SIM + '.pkl', 'rb'))
    except FileNotFoundError:
        gp = pystan.StanModel(model_code=model_code_dict[MODEL_TO_SIM], model_name=MODEL_TO_SIM)
        pickle.dump(gp, open("../stan_dumps/" + MODEL_TO_SIM + '.pkl', 'wb'))
    # do the HMC sampling (default 4 chains)
    fit = gp.sampling(data=stan_data)
    samps = fit.extract()
    # get model summary
    summary_dict = fit.summary()

    # ----------------------------Do the model diagnostics-------------------------------------------
    sum_df = pd.DataFrame(summary_dict['summary'],
                          columns=summary_dict['summary_colnames'],
                          index=summary_dict['summary_rownames'])
    # get mean predictions per new observation
    y_test = y_test.to_list()
    y_hat = []
    for i in range(len(y_test)):
        y_hat.append(sum_df.loc['y_pred[{}]'.format(i + 1)]['mean'])
    # calc interpretable (MAPE in [0,1])  error measure
    ape = []
    for i in range(len(y_hat)):
        if y_test[i] == 0:
            ape.append(np.abs(y_test[i] - y_hat[i]) / np.abs(y_test[i] + 1))
        else:
            ape.append(np.abs(y_hat[i] - y_test[i]) / np.abs(y_test[i]))
    print("MAPE:", np.mean(ape))

    # plot chains
    # _ = plot_trace()
    # retrieve parameters
    # alpha_fit = pois_samps['alpha']
    # sigma_fit = pois_samps['sigma']
    # length_scale = pois_samps['length_scale']
    # # plot results
    # plot_trace(alpha_fit, 'alpha', data['alpha'])
    # plt.show()
    # plot_trace(length_scale, 'length scale', data['length_scale'])
    # plt.show()
    # plot_trace(sigma_fit, 'sigma', data['sigma'])
    # plt.show()

    # get_pred_plot_with_conf(fit, 250, y_hat, y_test)
    # plt.show()
    # # plot
    # plt.scatter(df.loc[~df.index.isin(sample)].x, y_hat, c='r')
    # plt.scatter(df.loc[~df.index.isin(sample)].x, df.loc[~df.index.isin(sample)].y, c='b')
    # plt.show()
