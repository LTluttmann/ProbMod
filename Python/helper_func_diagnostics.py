import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_trace(param, param_name='parameter', sim_val=None):
    """
    plots the sampling trace and the posterior distribution of the specified parameter
    :param param: model summary of the specified parameter -> fit.summary()['param']
    :param param_name: name of the parameter as to appear on the plot
    :param sim_val: the true value of the parameter
    :return: plot of the results
    """

    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

    # Plotting
    plt.subplot(2, 1, 1)
    plt.plot(param)
    plt.xlabel('samples')
    plt.ylabel(param_name)
    plt.axhline(mean, color='r', lw=2, linestyle='--')
    plt.axhline(median, color='c', lw=2, linestyle='--')
    if sim_val:
        plt.axhline(sim_val, color='k', lw=2, linestyle='--')
    plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
    plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
    plt.title('Trace and Posterior Distribution for {}'.format(param_name))

    plt.subplot(2, 1, 2)
    plt.hist(param, 30, density=True)
    sns.kdeplot(param, shade=True)
    plt.xlabel(param_name)
    plt.ylabel('density')
    plt.axvline(mean, color='r', lw=2, linestyle='--', label='mean')
    plt.axvline(median, color='c', lw=2, linestyle='--', label='median')
    if sim_val:
        plt.axvline(sim_val, color='k', lw=2, linestyle='--', label='true val')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)

    plt.gcf().tight_layout()
    plt.legend()


def get_pred_plot_with_conf(fit, iters, mean_preds, test):
    """
    function to plot true realization from test data against predictions +
    confidence intervals. confidence intervals are generated by sampling from the
    posterior distribution of the target variable
    :param fit: stan output from sampling function
    :param iters: num of random draws from posterior
    :param mean_preds: y_hat (mean prediction)
    :param test: y_test (test data realizations of y)
    :return: plotting of the results
    """
    for _ in range(iters):
        preds = []
        for row in range(len(mean_preds)):
            lamb = fit['y_pred[{}]'.format(row + 1)]
            preds.append(np.random.choice(lamb))
        plo = plt.plot(preds, color='lightsteelblue', alpha=0.1)
    mean_pred = mean_preds
    true = test
    plo = plt.plot(mean_pred, label="Backest.")
    plo = plt.plot(true, label="True", linestyle="--", alpha=0.7, color='r')
    plt.ylim([0, 20])
    plo = plt.legend()
    return plo
