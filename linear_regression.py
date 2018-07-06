import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

cropdat = pd.read_pickle('/Users/john/Projects/corn_yield_pred/data/full_data.pickle')
fe = pd.get_dummies(cropdat.fips)
X = cropdat.iloc[:, 4:50]
y = cropdat['corn_yield']
y = np.log(y)


mod = sm.OLS(y, X).fit()
print(mod.summary())