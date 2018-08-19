print("Loading Libraries...")
import pandas as pd
import numpy as np
import math
import seaborn as sns
import scipy.stats as stats

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
# Estimate WLS and return RMSE
def felm_rmse(y, X, weights, y_test, X_test):
    mod = sm.WLS(y, X, weights = weights).fit()
    pred = mod.predict(X_test)
    res = (y_test[:] - pred.values)
    t_stat, p_val = stats.ttest_ind(mod.resid, res, equal_var=False)
    return (np.sqrt(np.mean(res**2)), t_stat)  
    
def nnetwork_rmse(y, X, y_test, X_test):
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
    
    t_stat, p_val = stats.ttest_ind(nn_train_res, nn_test_res, equal_var=False)
    
    return (np.sqrt(np.mean(nn_test_res**2)), t_stat)

# Group/Cluster K-fold CV
def gc_kfold_cv(data, group, begin, end):
    """
    Group/Cluster K-fold Cross-validation method
    data:     data to filter with 'trend'
    begin:    start of cluster
    end:      end of cluster
    """
    
    data['group'] = group
    
    # Filter test and train based on begin and end
    test = data[data['group'].isin(range(begin, end))]
    train = data[~data['group'].isin(range(begin, end))]
    
    # Return train and test    
    dfs = {}    
    tsets =[train, test]
    
    for i, val in enumerate([1, 2]):
         dfs[val] = tsets[i]
         
    return dfs

#--------------------------------------------


def felm_cv(regdata, group):
# Loop through 1-31 years with 5 groups in test set and 26 train set
    i = 1
    l = False
    retrmse = []
    rettstat = []
    while (l == False):
    
        tset = gc_kfold_cv(regdat, group, i, i + 4)
        y_train = tset[1].ln_corn_yield
        X_train = tset[1].drop(['ln_corn_yield', 'corn_acres'], 1)
        weights = tset[1].corn_acres
        y_test = tset[2].ln_corn_yield
        X_test = tset[2].drop(['ln_corn_yield', 'corn_acres'], 1)
        inrmse, t_stat = felm_rmse(y_train, X_train, weights, y_test, X_test)
        retrmse.append(inrmse)
        rettstat.append(t_stat)
        if i == 27:
            return(np.mean(retrmse), np.std(retrmse), np.mean(t_stat))
            l = True
        i += 1


def nn_cv(regdata, group):
    # Loop through 1-31 years with 5 groups in test set and 26 train set
        i = 1
        l = False
        retrmse = []
        rettstat = []
        while (l == False):
            print("Estimating NN CV:", i)
            print("Previous RMSE: ", nn_rmse, "Previous t-stat: ", nn_tstat)
            tset = gc_kfold_cv(regdat, group, i, i + 4)
            y_train = tset[1].ln_corn_yield
            X_train = tset[1].drop(['ln_corn_yield'], 1)
                    
            y_test = tset[2].ln_corn_yield
            X_test = tset[2].drop(['ln_corn_yield'], 1)
                    
                    # Scale data based on train set
            sc = StandardScaler()
            X_scale = sc.fit(X_train)
            X_train = X_scale.transform(X_train)
            X_test = X_scale.transform(X_test)
                    
                    # Array y train, test
            y_train = np.array(y_train).reshape(-1, 1)
            y_test = np.array(y_test).reshape(-1, 1)
                    
            nn_rmse, nn_tstat = nnetwork_rmse(y_train, X_train, y_test, X_test)
            
            retrmse.append(nn_rmse)
            rettstat.append(nn_tstat)
            if i == 27:
                return(np.mean(retrmse), np.std(retrmse), np.mean(rettstat))
                l = True
            i += 1

def demean_values(data, fe):
	'''
	Demean fixed-effect (fe) from values
    ...
    Arguments
    ---------
    data             : data to demean
                      Flat array or list with dependent variable
    fe               : fixed-effect from data
   
    -------
    yxd               : returned demean values
	'''
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
cropdat = pd.read_pickle('/Users/john/Projects/corn_yield_pred/data/full_data.pickle')

# Baseline Regression Cross-Validation
print("Estimating Baseline regression")
basedat = cropdat[['ln_corn_yield', 'trend', 'trend_sq', 'corn_acres']]
fe = pd.get_dummies(cropdat.fips)
regdat = pd.concat([basedat, fe], axis=1)
base_rmse, base_se, base_tstat = felm_cv(basedat, cropdat['trend'])
base_rmse
base_se
base_tstat

# Degree Day Regression Cross-Validation
print("Estimating Degree Day Regression")
dddat = cropdat[['ln_corn_yield', 'dday0_10C', 'dday10_30C', 'dday30C', 'prec', 'prec_sq', 'trend', 'trend_sq', 'corn_acres']]
fe = pd.get_dummies(cropdat.fips)
regdat = pd.concat([dddat, fe], axis=1)
ddreg_rmse, ddreg_se, ddreg_tstat = felm_cv(dddat, cropdat['trend'])


#--------------------------
# Neural Network
print("Estimating Neural Network")
regdat = cropdat[['ln_corn_yield', 'dday0_10C', 'dday10_30C', 'dday30C', 'prec', 'prec_sq', 'trend', 'trend_sq']]
regdat = demean_values(regdat, cropdat.fips)

nn_rmse, nn_se, nn_tstat = nn_cv(regdat, cropdat['trend'])

fdat = {'Regression': ['Baseline', 'Degree Day', 'Neural Network'], 
        'RMSE': [base_rmse, ddreg_rmse, nn_rmse],
        't-stat': [base_tstat, ddreg_tstat, nn_tstat]}

fdat = pd.DataFrame(fdat, columns = ['Regression', 'RMSE', 't-stat'])
fdat['change'] = (fdat['RMSE'] - fdat['RMSE'].iloc[0])/fdat['RMSE'].iloc[0]
fdat.to_pickle('/Users/john/Projects/corn_yield_pred/lm_vs_nn_results.pickle')

#---------------
# print results
print("---Results-------------------------------------------- ")
print("Baseline: ", base_rmse, "(RMSE)", base_tstat, "(t-stat)")
print("Degree Day: ", ddreg_rmse, "(RMSE)", ddreg_tstat, "(t-stat)")
print("Neural Network: ", nn_rmse, "(RMSE)", nn_tstat, "(t-stat)")
print("------------------------------------------------------ ")

#nn_rrmse = (nn_rmse - baseline_rmse)/baseline_rmse

# Plot
#rrmse = np.array([lfe_rrmse*-1, nn_rrmse*-1, -1*(nn_rmse - ddreg_rmse)/ddreg_rmse])
#plt.rcdefaults()
#fig, ax = plt.subplots()
#ax = ax.bar(['FE Model', 'NN Model', 'NN vs FE'], rrmse)
#plt.ylabel('% Reduced RMSE from Baseline Model')
#plt.show(ax)

#X = cropdat[['dday0_10C', 'dday10_30C', 'dday30C', 'prec', 'prec_sq', 'trend', 'trend_sq']]
#y = cropdat['ln_corn_yield']
# Transform features
##sc = StandardScaler()
#X_scale = sc.fit(X)
#X = X_scale.transform(X)
