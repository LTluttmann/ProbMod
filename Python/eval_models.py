import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helper_func_diagnostics import calc_mape, calc_median_ape, plot_trace
import arviz as az

STAN_MODEL_PATH = "../stan_dumps/"
MODEL_TO_EVAL = "pred_base"

if __name__ == "__main__":
    with open(STAN_MODEL_PATH + "fit_{}.pkl".format(MODEL_TO_EVAL), "rb") as f:
        data_dict = pickle.load(f)
    fit = data_dict['fit']
    y_test = data_dict['y_test']
    X_test = data_dict['X_test']
    summary_dict = fit.summary()
    sum_df = pd.DataFrame(summary_dict['summary'],
                          columns=summary_dict['summary_colnames'],
                          index=summary_dict['summary_rownames'])
    # get mean predictions per new observation
    y_hat = []
    for i in range(y_test.shape[0]):
        y_hat.append(sum_df.loc['y_pred[{}]'.format(i + 1)]['mean'])
    if MODEL_TO_EVAL == "pred_gp":
        y_hat = np.exp(y_hat)
    print("MAPE: ", calc_mape(y_hat, y_test))
    # print("MAPE: ", calc_mape(np.exp(y_hat), np.exp(y_test)))
    df_pred = pd.DataFrame(np.vstack((y_test, y_hat)).T, columns=["true", "pred"])
    df_pred.index = X_test.index
    df_pred = pd.concat([df_pred, X_test], axis=1)
    df_pred.to_pickle("../results/{}_df.pkl".format(MODEL_TO_EVAL))
    print(df_pred)
    pois_samps = fit.extract()
    print(pois_samps["beta"])
    #x = az.plot_trace(fit, var_names=["alpha", "beta"])
    #plt.show()
    # ----------------------------Do the model diagnostics-------------------------------------------
    # # retrieve parameters
    # alpha_fit = pois_samps['beta']
    # sigma_fit = pois_samps['sigma']
    # length_scale = pois_samps['length_scale']
    # # plot results
    # plot_trace(alpha_fit, 'alpha')
    # plt.show()
    # plot_trace(length_scale, 'length scale')
    # plt.show()
    # plot_trace(sigma_fit, 'sigma')
    # plt.show()
