print("Loading Libraries...")
import pandas as pd
import numpy as np
import math
import seaborn as sns
import scipy.stats as stats
import urllib

# Linear Regression w/ FE
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

# Plotting
import matplotlib.pyplot as plt

# Neural Network
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from sklearn.preprocessing import StandardScaler

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

    # Calculate ttest to check residuals from test and train are independent
    t_stat, p_val = stats.ttest_ind(mod.resid, res, equal_var=False)

    # Return RMSE and t-stat from ttest
    return (np.sqrt(np.mean(res**2)), t_stat) 
    

def nnetwork_rmse(y, X, y_test, X_test):
    """
    Estimate Neural Network and return tuple: RMSE and ttest of test and train residuals

    Keyword arguments:
    ------------------
    y:        Dep variable - Full or training data
    X:        Covariates - Full or training data
    y_test:   Dep variable - test data
    X_test:   Covariates - test data
    """
    # Build sequential model
    ksmod = Sequential()
    ksmod.add(Dense(40, input_dim=X.shape[1], activation='relu'))
    ksmod.add(Dense(20, activation='relu'))
    ksmod.add(Dense(10, activation='relu'))
    ksmod.add(Dense(1, activation='relu'))
    ksmod.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        
    # Fit model
    ksmod.fit(X, y, epochs=20, batch_size = 32)
    
    # Training residuals
    nn_train_pred = ksmod.predict(X)
    nn_train_res = (y[:] - nn_train_pred)
    
    # Test residuals
    nn_test_pred = ksmod.predict(X_test)
    nn_test_res = (y_test - nn_test_pred)
    
    # Calculate ttest to check that residuals from test and train are independent
    t_stat, p_val = stats.ttest_ind(nn_train_res, nn_test_res, equal_var=False)
    
    # Return RMSE and t-stat from ttest
    return (np.sqrt(np.mean(nn_test_res**2)), t_stat)


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
    data = data.assign(group=group.values)
    
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


def nn_cv(regdata, group):
    """
    Cross-validate Neural Network model and return RMSE and ttstat from ttest

    Keyword arguments:
    ------------------
    regdata:  regression data
    group:    group fixed effect
    """
    # Loop through 1-31 years with 5 groups in test set and 26 train set
    i = 1
    j = False
    retrmse = []
    rettstat = []
    nn_rmse = 0
    nn_tstat = 0
    while j == False:
        print("Estimating NN CV:", i)
        print("Previous RMSE: ", nn_rmse, "Previous t-stat: ", nn_tstat)

        # Get train and test data
        tset = gc_kfold_cv(regdata, group, i, i + 4)
        
        # Separate train and test data
        y_train = tset[1].ln_corn_yield
        X_train = tset[1].drop(['ln_corn_yield'], 1)
                
        y_test = tset[2].ln_corn_yield
        X_test = tset[2].drop(['ln_corn_yield'], 1)
                
        # Scale data based on train set
        sc = StandardScaler()
        X_scale = sc.fit(X_train)
        X_train = X_scale.transform(X_train)
        X_test = X_scale.transform(X_test)
                
        # Array y train, y test
        y_train = np.array(y_train).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)
                
        # Fit Neural Network and return RMSE and tstat
        nn_rmse, nn_tstat = nnetwork_rmse(y_train, X_train, y_test, X_test)
        
        # Append RMSE and tstat to return
        retrmse.append(nn_rmse)
        rettstat.append(nn_tstat)

        # If end of loop, return mean RMSE, s.e., and tstat
        if i == 27:
            return(np.mean(retrmse), np.std(retrmse), np.mean(rettstat))
            j = True
        i += 1


def demean_values(data, fe):
	"""
	Demean fixed-effect (fe) from data and return
    
    Keyword arguments:
    -----------------
    data   : data to demean
    fe     : fixed-effect from data
	"""
	yx = pd.DataFrame(data)
	yx['fe'] = fe
	demeaner1 = yx.groupby('fe').mean()
	demeaner1 = yx[['fe']].join(demeaner1, on='fe')\
    	    .drop(['fe'], axis=1)
	yx = yx.drop(['fe'], axis=1)
	yxd = yx - demeaner1	
	return yxd        

# Load data from build_data.py
print("Loading data.........")
#cropdat = pd.read_pickle('/Users/john/Projects/corn_yield_pred/data/full_data.pickle')

# Download remote from github
urllib.request.urlretrieve("https://github.com/johnwoodill/corn_yield_pred/raw/master/data/full_data.pickle", "full_data.pickle")
cropdat = pd.read_pickle("full_data.pickle")

# Baseline Regression Cross-Validation
print("Estimating Baseline regression")
basedat = cropdat[['ln_corn_yield', 'trend', 'trend_sq', 'corn_acres']]
fe = pd.get_dummies(cropdat.fips)
regdat = pd.concat([basedat, fe], axis=1)
base_rmse, base_se, base_tstat = felm_cv(basedat, cropdat['trend'])

# Degree Day Regression Cross-Validation
print("Estimating Degree Day Regression")
dddat = cropdat[['ln_corn_yield', 'dday0_10C', 'dday10_30C', 'dday30C', 'prec', 'prec_sq', 'trend', 'trend_sq', 'corn_acres']]
fe = pd.get_dummies(cropdat.fips)
regdat = pd.concat([dddat, fe], axis=1)
ddreg_rmse, ddreg_se, ddreg_tstat = felm_cv(regdat, cropdat['trend'])

#--------------------------
# Neural Network
print("Estimating Neural Network")
regdat = cropdat[['ln_corn_yield', 'dday0_10C', 'dday10_30C', 'dday30C', 'prec', 'prec_sq', 'trend', 'trend_sq']]
regdat = demean_values(regdat, cropdat.fips)

# Cross validate neural network
nn_rmse, nn_se, nn_tstat = nn_cv(regdat, cropdat['trend'])

# Get results as data.frame
fdat = {'Regression': ['Baseline', 'Degree Day', 'Neural Network'], 
        'RMSE': [base_rmse, ddreg_rmse, nn_rmse],
        't-stat': [base_tstat, ddreg_tstat, nn_tstat]}

fdat = pd.DataFrame(fdat, columns = ['Regression', 'RMSE', 't-stat'])

# Calculate percentage change
fdat['change'] = (fdat['RMSE'] - fdat['RMSE'].iloc[0])/fdat['RMSE'].iloc[0]

# Save
#fdat.to_pickle('/Users/john/Projects/corn_yield_pred/lm_vs_nn_results.pickle')

#---------------
# print results
print("---Results-------------------------------------------- ")
print("Baseline: ", base_rmse, "(RMSE)", base_tstat, "(t-stat)")
print("Degree Day: ", ddreg_rmse, "(RMSE)", ddreg_tstat, "(t-stat)")
print("Neural Network: ", nn_rmse, "(RMSE)", nn_tstat, "(t-stat)")
print("------------------------------------------------------ ")

# Plot
rrmse = np.array([fdat['change'].iloc[1]*-1, fdat['change'].iloc[2]*-1])
plt.rcdefaults()
fig, ax = plt.subplots()
ax = ax.bar(['FE Model', 'NN Model'], rrmse)
plt.ylabel('% Reduced RMSE from Baseline Model')
plt.show(ax)

