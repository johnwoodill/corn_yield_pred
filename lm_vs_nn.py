import pandas as pd
import numpy as np
import math
import seaborn as sns

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

# Demean variables
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

# Leave-one-out cross-validation
def loo_cv(mod, data):
	for i in range(pd.to_numeric(data.trend)):
		train = data[data.trend != i]

# Estimate WLS and return RMSE
def felm_rmse(y, X, weights, X_test, y_test):
    mod = sm.WLS(y, X, weights = weights).fit()
    pred = mod.predict(X_test)
    res = (np.array(y_test) - np.array(pred))**2
    return np.sqrt(np.mean(res))
    
def nnetwork_rmse(y, X, test_data):
    # Build sequential model
    ksmod = Sequential()
    ksmod.add(Dense(12, input_dim=X.shape[1], activation='relu'))
    ksmod.add(Dense(8, activation='relu'))
    ksmod.add(Dense(4, activation='relu'))
    ksmod.add(Dense(1, activation='relu'))
    ksmod.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    
    # Fit model
    ksmod.fit(X, y, epochs=10, batch_size = 20)
    nn_pred = ksmod.predict(test_data)
    
    # Get RMSE
    nn_res = (np.array(y[['ln_corn_yield']]) - np.array(nn_pred))**2
    return np.sqrt(np.mean(nn_res))


# Load data from build_data.py
cropdat = pd.read_pickle('/Users/john/Projects/corn_yield_pred/data/full_data.pickle')

#-------------------------
# Baseline regression
X = cropdat[['trend', 'trend_sq']]
y = cropdat[['ln_corn_yield']]

# Get dummy FE, combine,
fe = pd.get_dummies(cropdat.fips)
X = pd.DataFrame(np.hstack((X, fe)))

# Get RMSE
baseline_rmse = felm_rmse(y, X, weights = cropdat[['corn_acres']], X_test = X, y_test = y)
baseline_rmse

#---------------------------------
# Standard Degree Day Regression
X = cropdat[['dday0_10C', 'dday10_30C', 'dday30C', 'prec', 'prec_sq', 'trend', 'trend_sq']]
y = cropdat[['ln_corn_yield']]
fe = pd.get_dummies(cropdat.fips)
X = pd.DataFrame(np.hstack((X, fe)))

# Get RMSE
ddreg_rmse = felm_rmse(y, X, weights = cropdat[['corn_acres']], test_data = X)
ddreg_rmse

lfe_rrmse = (ddreg_rmse - baseline_rmse)/ddreg_rmse
lfe_rrmse

#--------------------------
# Neural Network
X = cropdat[['corn_yield', 'dday0_10C', 'dday10_30C', 'dday30C', 'prec', 'prec_sq', 'trend', 'trend_sq']]

# Transform features
sc = StandardScaler()
X_scale = sc.fit(X)
X = X_scale.transform(X)

y = np.array(y).reshape(-1, 1)

nn_rmse = nnetwork_rmse(y, X, test_data = X)

# Build sequential model
ksmod = Sequential()
ksmod.add(Dense(12, input_dim=X.shape[1], activation='relu'))
ksmod.add(Dense(8, activation='relu'))
ksmod.add(Dense(4, activation='relu'))
ksmod.add(Dense(1, activation='relu'))
ksmod.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Fit model
ksmod.fit(X, y, epochs=50, batch_size = 32)
nn_pred = ksmod.predict(test)

# Get RMSE
nn_errors = y - nn_pred
nn_rmse = np.sqrt(np.mean(nn_errors**2))
nn_rrmse = (nn_rmse - baseline_rmse)/baseline_rmse
nn_rrmse

rrmse = np.array([lfe_rrmse*-1, nn_rrmse*-1])

# Plot
plt.rcdefaults()
fig, ax = plt.subplots()
ax = ax.bar(['FE Model', 'NN Model'], rrmse)
plt.ylabel('% Reduced RMSE from Baseline Model')
plt.show(ax)


