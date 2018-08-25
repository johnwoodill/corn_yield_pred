# -*- coding: utf-8 -*-
"""
Code Summary:

This script uses a custom data set to cross-validate a baseline regression that
estimates corn yields with fixed effects and trends. A second regression is
cross-validated that includes degree days and precipitation to show
percentage change improvements in RMSE from baseline. Additionally, a custom
function splits data group into clusters of 5 years. A t-test is used to
validate the residual errors are independent between test and train data.

Data description:

Corn yield data was downloaded from NASS between 1980-2010 for the US corn belt
states: Indiana, Illinois, Iowa. Temperature data -- degree days and precip --
are provided from Schlenker and Roberts (2009).

URL: "https://github.com/johnwoodill/corn_yield_pred/raw/master/data/full_data.pickle"

Example:
    The data is automatically downloaded and processed within the script,
    so a simple python call is all that is needed.

    $ python cv_lm.py
"""
from urllib import request
from scipy import stats
import pandas as pd
import numpy as np
import statsmodels.api as sm

print("Defining functions......")

def main():
    """
    Estimate baseline and degree day regression.

    Returns:
        data.frame with RMSE, SE, and tstats
    """
    # Download remote from github
    print("Downloading custom data set from: ")
    print("https://github.com/johnwoodill/corn_yield_pred/raw/master/data/full_data.pickle")
    file_url = "https://github.com/johnwoodill/corn_yield_pred/raw/master/data/full_data.pickle"
    request.urlretrieve(file_url, "full_data.pickle")
    cropdat = pd.read_pickle("full_data.pickle")

    # Baseline WLS Regression Cross-Validation with FE and trends
    print("Estimating Baseline Regression")
    basedat = cropdat[['ln_corn_yield', 'trend', 'trend_sq', 'corn_acres']]
    fe_group = pd.get_dummies(cropdat.fips)
    regdat = pd.concat([basedat, fe_group], axis=1)
    base_rmse, base_se, base_tstat = felm_cv(regdat, cropdat['trend'])

    # Degree Day Regression Cross-Validation
    print("Estimating Degree Day Regression")
    dddat = cropdat[['ln_corn_yield', 'dday0_10C', 'dday10_30C', 'dday30C',
                     'prec', 'prec_sq', 'trend', 'trend_sq', 'corn_acres']]
    fe_group = pd.get_dummies(cropdat.fips)
    regdat = pd.concat([dddat, fe_group], axis=1)
    ddreg_rmse, ddreg_se, ddreg_tstat = felm_cv(regdat, cropdat['trend'])

    # Get results as data.frame
    fdat = {'Regression': ['Baseline', 'Degree Day',],
            'RMSE': [base_rmse, ddreg_rmse],
            'se': [base_se, ddreg_se],
            't-stat': [base_tstat, ddreg_tstat]}

    fdat = pd.DataFrame(fdat, columns=['Regression', 'RMSE', 'se', 't-stat'])

    # Calculate percentage change
    fdat['change'] = (fdat['RMSE'] - fdat['RMSE'].iloc[0])/fdat['RMSE'].iloc[0]
    return fdat


def felm_rmse(y_train, x_train, weights, y_test, x_test):
    """
    Estimate WLS from y_train, x_train, predict using x_test, calculate RMSE,
    and test whether residuals are independent.

    Arguments:
        y_train: Dep variable - Full or training data
        x_train: Covariates - Full or training data
        weights: Weights for WLS
        y_test: Dep variable - test data
        x_test: Covariates - test data

    Returns:
        Returns tuple with RMSE and tstat from ttest
    """
    # Fit model and get predicted values of test data
    mod = sm.WLS(y_train, x_train, weights=weights).fit()
    pred = mod.predict(x_test)

    #Get residuals from test data
    res = (y_test[:] - pred.values)

    # Calculate ttest to check residuals from test and train are independent
    t_stat = stats.ttest_ind(mod.resid, res, equal_var=False)[0]

    # Return RMSE and t-stat from ttest
    return (np.sqrt(np.mean(res**2)), t_stat)

def gc_kfold_cv(data, group, begin, end):
    """
    Custom group/cluster data split for cross-validation of panel data.
    (Ensure groups are clustered and train and test residuals are independent)

    Arguments:
        data:     data to filter with 'trend'
        group:    group to cluster
        begin:    start of cluster
        end:      end of cluster

    Return:
        Return test and train data for Group-by-Cluster Cross-validation method
    """
    # Get group data
    data = data.assign(group=group.values)

    # Filter test and train based on begin and end
    test = data[data['group'].isin(range(begin, end))]
    train = data[~data['group'].isin(range(begin, end))]

    # Return train and test
    dfs = {}
    tsets = [train, test]

    # Combine train and test to return dfs
    for i, val in enumerate([1, 2]):
        dfs[val] = tsets[i]

    return dfs

def felm_cv(regdata, group):
    """
    Cross-validate WLS FE model

    Arguments:
        regdata:  regression data
        group:    group fixed effect

    Returns:
        return mean RMSE, standard error, and mean tstat from ttest
    """
    # Loop through 1-31 years with 5 groups in test set and 26 train set
    #i = 1
    #j = False
    retrmse = []
    rettstat = []
    #for j, val in enumerate([1, 27]):
    for j in range(1, 28):
        # Get test and training data
        tset = gc_kfold_cv(regdata, group, j, j + 4)

        # Separate y_train, x_train, y_test, x_test, and weights
        y_train = tset[1].ln_corn_yield
        x_train = tset[1].drop(['ln_corn_yield', 'corn_acres'], 1)
        weights = tset[1].corn_acres
        y_test = tset[2].ln_corn_yield
        x_test = tset[2].drop(['ln_corn_yield', 'corn_acres'], 1)

        # Get RMSE and tstat from train and test data
        inrmse, t_stat = felm_rmse(y_train, x_train, weights, y_test, x_test)

        # Append RMSE and tstats to return
        retrmse.append(inrmse)
        rettstat.append(t_stat)

        # If end of loop return mean RMSE, s.e., and tstat
        if j == 27:
            return (np.mean(retrmse), np.std(retrmse), np.mean(t_stat))

if __name__ == "__main__":
    RDAT = main()
    print(RDAT)

    # print results
    print("---Results--------------------------------------------")
    print("Baseline: ", round(RDAT.iloc[0, 1], 2), "(RMSE)",
          round(RDAT.iloc[0, 2], 2), "(se)",
          round(RDAT.iloc[0, 1], 3), "(t-stat)")
    print("Degree Day: ", round(RDAT.iloc[1, 1], 2), "(RMSE)",
          round(RDAT.iloc[0, 2], 2), "(se)",
          round(RDAT.iloc[1, 3], 2), "(t-stat)")
    print("------------------------------------------------------")
    print("% Change from Baseline: ", round(RDAT.iloc[1, 4], 4)*100, "%")
    print("------------------------------------------------------")
