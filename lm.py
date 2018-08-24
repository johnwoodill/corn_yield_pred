print("Loading Libraries...")
import pandas as pd
import numpy as np
import scipy.stats as stats
import urllib
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

print("Defining functions......")

def felm_rmse(y, X, weights, y_test, X_test):
    """
    Estimate WLS and return tuple: RMSE and ttest of test and train residuals

    Keyword arguments:
    ------------------
    y:        Dep variable - Full or training data
    X:        Covariates - Full or training data
    weights:  Weights for WLS
    y_test:   Dep variable - test data
    X_test:   Covariates - test data
    """
    # Fit model and get predicted values of test data
    mod = sm.WLS(y, X, weights = weights).fit()
    pred = mod.predict(X_test)
    
    # Get residuals from test data
    res = (y_test[:] - pred.values)

    # Calculate ttest to check that residuals from test and train are independent
    t_stat, p_val = stats.ttest_ind(mod.resid, res, equal_var=False)

    # Return RMSE and t-stat from ttest
    return (np.sqrt(np.mean(res**2)), t_stat)  
    
def gc_kfold_cv(data, group, begin, end):
    """
    Return test and train data for Group-by-Cluster Cross-validation method
    (Need to ensure groups are clustered and train and test residuals are independent)

    Keyword arguments:
    ------------------
    data:     data to filter with 'trend'
    group:    group to cluster
    begin:    start of cluster
    end:      end of cluster
    """
    # Get group data
    data['group'] = group
    
    # Filter test and train based on begin and end
    test = data[data['group'].isin(range(begin, end))]
    train = data[~data['group'].isin(range(begin, end))]
    
    # Return train and test    
    dfs = {}    
    tsets =[train, test]
    
    # Combine train and test to return dfs
    for i, val in enumerate([1, 2]):
        dfs[val] = tsets[i]
         
    return dfs



def felm_cv(regdata, group):
    """
    Cross-validate WLS FE model and return RMSE and ttstat from ttest

    Keyword arguments:
    ------------------
    regdata:  regression data
    group:    group fixed effect
    """
    # Loop through 1-31 years with 5 groups in test set and 26 train set
    i = 1
    l = False
    retrmse = []
    rettstat = []
    while (l == False):
        # Get test and training data
        tset = gc_kfold_cv(regdat, group, i, i + 4)
        
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
            l = True

        # If not end of loop increase one
        i += 1

# Download remote from github
urllib.request.urlretrieve("https://github.com/johnwoodill/corn_yield_pred/raw/master/data/full_data.pickle", "full_data.pickle")
cropdat = pd.read_pickle("full_data.pickle")

# Degree Day Regression Cross-Validation
print("Estimating Degree Day Regression")
dddat = cropdat[['ln_corn_yield', 'dday0_10C', 'dday10_30C', 'dday30C', 'prec', 'prec_sq', 'trend', 'trend_sq', 'corn_acres']]
fe = pd.get_dummies(cropdat.fips)
regdat = pd.concat([dddat, fe], axis=1)
ddreg_rmse, ddreg_se, ddreg_tstat = felm_cv(dddat, cropdat['trend'])

print("Degree Day: ", ddreg_rmse, "(RMSE)", ddreg_se, "(SE)",  ddreg_tstat, "(t-stat)")