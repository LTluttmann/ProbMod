import numpy as np
import pandas as pd
import pystan
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler, FunctionTransformer

# import of stan models
from gp_pr_stan import gp_mod_pois
from gp_stan import gp_mod
from orig_pois_reg_stan import hierarchical_model
from regularized_pois_reg_stan import hierarchical_model_reg

# ---------------------------------------------------- CONFIG ----------------------------------------------------------
# path to the data
DATAMART_PATH = '../Data/final.xlsx'
SENT_PATH = '../Data/sent_scores_df.pkl'
# define path to store compiled stan model
STAN_MODEL_PATH = "../stan_dumps/"
#
MODELS_TO_TRAIN = ["pred_base"]  # OPTIONS: pred_gp, pred_gp_pr, pred_base, pred_horseshoe

DUMP_COMPILED_MODEL = True
# Define Target variable and Features
TARGET = "worldwide_box_office"
FEATURES = [
    'production_budget', 'dist', 'stars', 'direc',
    'opening_weekend_revenue', 'num_pos_tweets',
    'adventure', 'comedy', 'docu', 'drama', 'horror', 'musical', 'thriller',
    'action'
]
""""""
# ----------------------------------------------------------------------------------------------------------------------

model_code_dict = dict(
    pred_gp=gp_mod,
    pred_gp_pr=gp_mod_pois,
    pred_base=hierarchical_model,
    pred_horseshoe=hierarchical_model_reg
)


def get_and_filter_df(path_to_movie_data, path_to_sent_analysis):
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
    #final_df = final_df.loc[~pd.Series((np.log(final_df.production_budget) > 2.75) & (final_df.screens < 1000))]
    # remove outlier (hard to predict with any model)
    #up_quant = np.quantile(final_df[TARGET], 0.85)
    #low_quant = np.quantile(final_df[TARGET], 0.05)
    #final_df = final_df.loc[(final_df[TARGET] > low_quant) & (final_df[TARGET] < up_quant)]
    return final_df


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


def get_model_data_dict(X_train, X_test, y_train, scale=True):
    X_train=X_train.copy()
    X_test=X_test.copy()
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        X_test = X_test.values
    stan_data_gp = dict(N=X_train.shape[0], D=X_train.shape[1], N_pred=X_test.shape[0],
                        zeros=np.zeros(X_train.shape[0]), x=X_train,
                        y=np.log(y_train.values.ravel()),
                        x_pred=X_test)

    stan_data_gp_pr = dict(N=X_train.shape[0], D=X_train.shape[1], N_pred=X_test.shape[0],
                           zeros=np.zeros(X_train.shape[0]), x=X_train,
                           y=y_train.values.ravel(),
                           x_pred=X_test)

    if not scale:
        transformer = FunctionTransformer(np.log1p, validate=True)
        X_train = transformer.transform(X_train)
        X_test = transformer.transform(X_test)

    stan_data_base = dict(x=X_train, N=X_train.shape[0], C=1, D=X_train.shape[1],
                          c=[0] * X_train.shape[0], y=y_train.values.ravel(),
                          N_test=X_test.shape[0], x_test=X_test, c_test=[0] * X_test.shape[0])

    mod_data_dict = dict(
        pred_gp=stan_data_gp,
        pred_gp_pr=stan_data_gp_pr,
        pred_base=stan_data_base,
        pred_horseshoe=stan_data_base
    )
    return mod_data_dict


if __name__ == "__main__":
    np.random.seed(652823193)
    df = get_and_filter_df(DATAMART_PATH, SENT_PATH)
    X_train, X_test, y_train, y_test = get_train_test(df, features=FEATURES, frac_test_set=0.2, scale_factor=1000, sample_size=150)
    model_data_dict = get_model_data_dict(X_train, X_test, y_train, scale=False)
    for mod in MODELS_TO_TRAIN:
        print("Now doing inference with model {}".format(mod))
        try:
            model = pickle.load(open(STAN_MODEL_PATH + mod + '.pkl', 'rb'))
        except FileNotFoundError:
            model = pystan.StanModel(model_code=model_code_dict[mod], model_name=mod)
            if DUMP_COMPILED_MODEL:
                pickle.dump(model, open(STAN_MODEL_PATH + mod + '.pkl', 'wb'))
        model_data = model_data_dict[mod]
        fit = model.sampling(model_data, n_jobs=4, chains=4, iter=3000)
        with open(STAN_MODEL_PATH + "fit_{}.pkl".format(mod), "wb") as f:
            pickle.dump({'model': model, 'fit': fit, 'y_test': y_test, 'X_test': X_test}, f, protocol=-1)
