import numpy as np
import pandas as pd
import pystan
import pickle
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from helper_func_diagnostics import calc_mape, calc_rmse
from time import time
# import of stan models
from stan.gp_pr_stan import gp_mod_pois, hierarchical_gp_pr_mod
from stan.gp_stan import gp_mod, hierarchical_gp_mod
from stan.orig_pois_reg_stan import hierarchical_model_pois
from stan.regularized_pois_reg_stan import hierarchical_model_regularized

# ---------------------------------------------------- CONFIG ----------------------------------------------------------
# path to the data
DATAMART_PATH = '../Data/final.xlsx'
SENT_PATH = '../Data/sent_scores_df.pkl'

# define path to store compiled stan model
STAN_MODEL_PATH = "../stan_dumps/"
PRED_RESULTS_PATH = "../results/"

# specify the models to be trained -- OPTIONS:
# pred_gp: gaussian process regression with normal output
# pred_gp_pr: gaussian process regression with poisson output
# pred_base: baseline model by Neelemegham
# pred_horseshoe: baseline with shrinkage priors on regression weights
# pred_hiera_gp_pr : gaussian process regression with normal output and hierarchies
# pred_hiera_gp: gaussian process regression with poisson output and hierarchies
MODELS_TO_TRAIN = ["pred_base", "pred_horseshoe", "pred_gp"]

# Define Target variable
TARGET = "worldwide_box_office"

# Features to be used in model training / prediction
FEATURES = [
    'production_budget', 'dist', 'stars', 'direc', 'screens',
    'opening_weekend_revenue', 'ratio_pos_tweets', 'total_tweets',
    'adventure', 'comedy', 'docu', 'drama', 'horror', 'musical', 'thriller',
    'action'
]
""""""
# ----------------------------------------------------------------------------------------------------------------------

# assign the stan code to their corresponding models
model_code_dict = dict(
    pred_gp=gp_mod,
    pred_gp_pr=gp_mod_pois,
    pred_base=hierarchical_model_pois,
    pred_horseshoe=hierarchical_model_regularized,
    pred_hiera_gp_pr=hierarchical_gp_pr_mod,
    pred_hiera_gp=hierarchical_gp_mod
)

# very large target variables lead to issues with poisson regression (max. lambda)
scales = dict(
    pred_gp="log",
    pred_gp_pr=10000,
    pred_base=10000,
    pred_horseshoe=10000,
    pred_hiera_gp_pr=10000,
    pred_hiera_gp='log'
)


def get_and_filter_df(path_to_movie_data: str, path_to_sent_analysis: str):
    """
    Function retrieves the dataset that has been build in a separate R file. Furthermore, retrieves the
    results of the sentiment analysis and joins it to orginal data frame. Finally, severe outliers are removed
    :param path_to_movie_data: Path to movie data
    :param path_to_sent_analysis: Path to results of sentiment analysis
    :return: data frame containing input space variables and target variable
    """
    df_mdb = pd.read_excel(path_to_movie_data)
    df_mdb.columns = [x.lower().replace(" ", "_") for x in df_mdb.columns]
    df_mdb.set_index('title', inplace=True)
    df_sent = pd.read_pickle(path_to_sent_analysis)
    df_sent['num_pos_tweets'] = df_sent.ratio_pos_tweets * df_sent.total_tweets
    final_df = df_mdb.merge(df_sent, left_index=True, right_index=True, how='inner')
    # exclude any missing data
    final_df = final_df.dropna(axis=0)
    # Production budget of zero makes no sense, probably missing data
    final_df = final_df[final_df.production_budget > 0]
    # EDA (see script EDA.ipynb) has shown that movies with the following conditions are severe outliers
    # (probably mistakes in data)
    final_df = final_df.loc[~pd.Series((np.log(final_df.production_budget) > 2.75) & (final_df.screens < 1000))]
    # remove outlier (hard to predict with any model)
    up_quant = np.quantile(final_df[TARGET], 0.95)
    low_quant = np.quantile(final_df[TARGET], 0.05)
    final_df = final_df.loc[(final_df[TARGET] > low_quant) & (final_df[TARGET] < up_quant)]
    return final_df


def get_model_data_dict(model_class: str, train_df_x: pd.DataFrame, test_df_x: pd.DataFrame,
                        train_df_y: pd.Series, C:int=None, c_ind=None, c_test_ind=None):
    """
    This function sets up the data that will be passed to the model specified in model_class. Moreover, depending on
    the model class, the input data is scaled. For the baseline model and the regularized extension of it, the input
    data is log transformed as proposed by Neelemegham and Chintagunta (1999). For Gaussian Processes, a MinMax scale
    is performed in order to squish all variables on the same scale (otherwise features on large scale have the largest
    impact)
    :param model_class: string specifying the model to be trained
    :param train_df_x: pandas data frame containing the input vectors of the training data set
    :param test_df_x: pandas data frame containing the input vectors of the test data set
    :param train_df_y: pandas Series containing the target variable of the training data set
    :param C: number of different countries / groups
    :param c_ind: list of size of training data set, containing the countries of the respective row in X_train
    :param c_test_ind: list of size of test data set, containing the countries of the respective row in X_test
    :return: input for stan model containing the data specified in the data block of the model
    """
    train_df_x = train_df_x.copy()
    test_df_x = test_df_x.copy()
    if model_class in ['pred_gp_pr', 'pred_gp']:
        # standard scale to avoid biases in rbf kernel
        scaler = MinMaxScaler()
        train_df_x = scaler.fit_transform(train_df_x)
        test_df_x = scaler.transform(test_df_x)
        stan_data_gp = dict(N=train_df_x.shape[0], D=train_df_x.shape[1], N_pred=test_df_x.shape[0],
                            zeros=np.zeros(train_df_x.shape[0]), x=train_df_x,
                            y=train_df_y.ravel(), x_pred=test_df_x)
        return stan_data_gp

    elif model_class in ['pred_base', 'pred_horseshoe']:
        # log transformation first
        transformer = FunctionTransformer(np.log1p, validate=True)
        train_df_x = transformer.transform(train_df_x)
        test_df_x = transformer.transform(test_df_x)
        stan_data_base = dict(x=train_df_x, N=train_df_x.shape[0], C=C, D=train_df_x.shape[1],
                              c=c_ind, y=train_df_y.values.ravel(),
                              N_test=test_df_x.shape[0], x_test=test_df_x, c_test=c_test_ind)
        return stan_data_base

    elif model_class in ['pred_hiera_gp_pr', 'pred_hiera_gp']:
        # standard scale to avoid biases in rbf kernel
        scaler = StandardScaler()
        train_df_x = scaler.fit_transform(train_df_x)
        test_df_x = scaler.transform(test_df_x)
        stan_data_hiera_gp = dict(N=train_df_x.shape[0], D=train_df_x.shape[1], N_pred=test_df_x.shape[0],
                                  zeros=np.zeros(train_df_x.shape[0]), x=train_df_x,
                                  y=train_df_y.ravel(), x_pred=test_df_x, C=C, c_ind=c_ind,
                                  c_ind_test=c_test_ind)
        return stan_data_hiera_gp
    else:
        raise ValueError("Wrong model class specified")


if __name__ == "__main__":
    np.random.seed(15214)
    df = get_and_filter_df(DATAMART_PATH, SENT_PATH)
    N_samps = [15, 20, 30, 50, 80, 120, 180]
    N_test = 40
    # first sample test datapoints, as those shall be the same for every experiment (e.g. for varying n)
    test_samps = np.random.choice(df.index, N_test, replace=False)
    for N in N_samps:
        train_samps = np.random.choice(list(set(df.index).difference(test_samps)), N, replace=False)
        X_train = df.loc[train_samps][FEATURES]
        X_test = df.loc[test_samps][FEATURES]
        y_train = df.loc[train_samps][TARGET]
        y_test = df.loc[test_samps][TARGET]
        for mod in MODELS_TO_TRAIN:
            # poisson regression not appropriate for very large target observations, rescaling might be necessary
            y_train_mod = y_train.copy()
            y_test_mod = y_test.copy()
            if type(scales[mod]) == int:
                y_train_mod = np.ceil(y_train / scales[mod]).astype('int')
                y_test_mod = np.ceil(y_test / scales[mod]).astype('int')
            elif scales[mod] == 'log':
                y_train_mod = np.log(y_train)
                y_test_mod = np.log(y_test)
            print("Now doing inference with model {}".format(mod))
            # load compiled model or compile stan code
            try:
                model = pickle.load(open(STAN_MODEL_PATH + mod + '.pkl', 'rb'))
            except FileNotFoundError:
                model = pystan.StanModel(model_code=model_code_dict[mod], model_name=mod)
                pickle.dump(model, open(STAN_MODEL_PATH + mod + '.pkl', 'wb'))
            # get data for stan model
            model_data = get_model_data_dict(mod, X_train, X_test, y_train_mod, C=1,
                                             c_ind=[1] * X_train.shape[0], c_test_ind=[1] * X_test.shape[0])
            # perform the MCMC and summarize results
            starttime = time()
            fit = model.sampling(model_data, n_jobs=3, chains=3, iter=2000, seed=246412)
            runtime = time() - starttime
            summary_dict = fit.summary()
            sum_df = pd.DataFrame(summary_dict['summary'],
                                  columns=summary_dict['summary_colnames'],
                                  index=summary_dict['summary_rownames'])
            # get mean predictions per new observation
            y_hat = [sum_df.loc['y_pred[{}]'.format(i + 1)]['mean'] for i in range(y_test_mod.shape[0])]
            if scales[mod] == 'log':
                y_hat = np.exp(y_hat)
                y_test_mod = np.exp(y_test_mod)
            elif type(scales[mod] == int):
                y_hat = np.array(y_hat) * scales[mod]
                y_test_mod = y_test_mod * scales[mod]
            print("MAPE for model {}: ".format(mod), calc_mape(y_hat, y_test_mod))
            print("RMSE for model {}: ".format(mod), calc_rmse(y_hat, y_test_mod))
            with open(PRED_RESULTS_PATH + "fit_{}_{}.pkl".format(mod, str(N)), "wb") as f:
                pickle.dump({'model': model, 'fit': fit, 'y_hat': y_hat, 'y_test': y_test_mod, 'X_test': X_test,
                             'runtime': runtime}, f, protocol=-1)
