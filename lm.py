# -*- coding: utf-8 -*-
"""

Code Summary:

This script uses a custom data set to cross-validate a baseline regression that
estimates corn yields with fixed effects and trends. A second regression is 
cross-validated that includes degree days and precipitation to show 
percentage change improvements in RMSE from baseline. Addtionally, a custom 
function splits data group into clusters of 5 years. A t-test is used to 
validate the residual errors are independent between test and train data.

Data description:

Corn yield data downloaded from NASS between 1980-2010 for the US corn belt 
states: Indiana, Illinois, Iowa. Temperature data -- degree days and precip --
is provided from Schlenker and Roberts (2009).

URL: "https://github.com/johnwoodill/corn_yield_pred/raw/master/data/full_data.pickle"

Example:
    The data is automatically downloaded and processed within the script,
    so a simple python call is all that is needed.

    $ python lm.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.
"""

import urllib
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

print("Defining functions......")

def felm_rmse(y, X, weights, y_test, X_test):
    """
    Estimate WLS from y, X, predict using X_test, calculate RMSE,
    and test whether residuals are independent.

    Arguments:
        y: Dep variable - Full or training data
        X: Covariates - Full or training data
        weights: Weights for WLS
        y_test: Dep variable - test data
        X_test: Covariates - test data

    Returns:
        Returns tuple with RMSE and tstat from ttest
    """
    # Fit model and get predicted values of test data
    mod = sm.WLS(y, X, weights=weights).fit()
    pred = mod.predict(X_test)

    #Get residuals from test data
    res = (y_test[:] - pred.values)

    # Calculate ttest to check that residuals from test and train are independent
    t_stat, p_val = stats.ttest_ind(mod.resid, res, equal_var=False)

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
    data['group'] = group
    
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
    i = 1
    j = False
    retrmse = []
    rettstat = []
    while j == False:
        # Get test and training data
        tset = gc_kfold_cv(regdata, group, i, i + 4)
        
        # Separate y_train, X_train, y_test, X_test, and weights
        y_train = tset[1].ln_corn_yield
        X_train = tset[1].drop(['ln_corn_yield', 'corn_acres'], 1)
        weights = tset[1].corn_acres
        y_test = tset[2].ln_corn_yield
        X_test = tset[2].drop(['ln_corn_yield', 'corn_acres'], 1)
        
        # Get RMSE and tstat from train and test data
        inrmse, t_stat = felm_rmse(y_train, X_train, weights, y_test, X_test)

        # Append RMSE and tstats to return
        retrmse.append(inrmse)
        rettstat.append(t_stat)
        
        # If end of loop return mean RMSE, s.e., and tstat
        if i == 27:
            return(np.mean(retrmse), np.std(retrmse), np.mean(t_stat))
            j = True

        # If not end of loop increase one
        i += 1

# Download remote from github
file_url = "https://github.com/johnwoodill/corn_yield_pred/raw/master/data/full_data.pickle"
urllib.request.urlretrieve(file_url, "full_data.pickle")
cropdat = pd.read_pickle("full_data.pickle")

# Baseline Regression Cross-Validation
print("Estimating Baseline Regression")
basedat = cropdat[['ln_corn_yield', 'trend', 'trend_sq', 'corn_acres']]
fe = pd.get_dummies(cropdat.fips)
regdat = pd.concat([basedat, fe], axis=1)
base_rmse, base_se, base_tstat = felm_cv(basedat, cropdat['trend'])

# Degree Day Regression Cross-Validation
print("Estimating Degree Day Regression")
dddat = cropdat[['ln_corn_yield', 'dday0_10C', 'dday10_30C', 'dday30C', 
                 'prec', 'prec_sq', 'trend', 'trend_sq', 'corn_acres']]
fe = pd.get_dummies(cropdat.fips)
regdat = pd.concat([dddat, fe], axis=1)
ddreg_rmse, ddreg_se, ddreg_tstat = felm_cv(dddat, cropdat['trend'])

# Get results as data.frame
fdat = {'Regression': ['Baseline', 'Degree Day',], 
        'RMSE': [base_rmse, ddreg_rmse],
        't-stat': [base_tstat, ddreg_tstat]}

fdat = pd.DataFrame(fdat, columns=['Regression', 'RMSE', 't-stat'])

# Calculate percentage change
fdat['change'] = (fdat['RMSE'] - fdat['RMSE'].iloc[0])/fdat['RMSE'].iloc[0]

# print results
print("---Results-------------------------------------------- ")
print("Baseline: ", round(base_rmse, 2), "(RMSE)", 
                    round(base_tstat, 2), "(t-stat)")
print("Degree Day: ", round(ddreg_rmse, 2), "(RMSE)", 
                      round(ddreg_tstat, 2), "(t-stat)")
print("------------------------------------------------------ ")
print("% Change from Baseline: ", round(fdat['change'].iloc[1], 2)*100, "%")
print("------------------------------------------------------ ")

