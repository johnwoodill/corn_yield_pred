import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# Time in each degree data
readRDS = robjects.r['readRDS']
dd = readRDS('/Users/john/Projects/corn_yield_pred/data/fips_degree_time_1900-2013.rds')
dd = pandas2ri.ri2py(dd)
dd = pd.DataFrame(dd)
dd['fips'] = dd['fips'].astype(int)
dd['year'] = dd['year'].astype(int)
dd['fips'] = dd['fips'].astype(str)
dd['year'] = dd['year'].astype(str)

# Degree days
dday = pd.read_stata('/Users/john/Projects/corn_yield_pred/data/FULL_ddayByYearandFips_cropAreaWeighted.dta')
dday['dday0_10C'] = dday['dday0C'] - dday['dday10C']
dday['dday10_30C'] = dday['dday10C'] - dday['dday30C']
dday['fips'] = dday['fips'].astype(int)
dday['year'] = dday['year'].astype(int)
dday['fips'] = dday['fips'].astype(str)
dday['year'] = dday['year'].astype(str)

# Precipitation
#prec = pd.read_csv('/Users/john/Projects/corn_yield_pred/data/fips_precipitation_1900-2013.csv')
#prec = prec[prec['month'] >= 3]
#prec = prec[prec['month'] <= 10]
#prec = prec.groupby(['fips', 'year'])['ppt'].sum()
#prec = prec.reset_index()
#prec['fips'] = prec['fips'].astype(str)
#prec['year'] = prec['year'].astype(str)
#prec['prec_sq'] = prec['ppt']**2
#prec.columns = ['fips', 'year', 'prec', 'prec_sq']

# Corn data
cropdat = pd.read_csv('/Users/john/Projects/corn_yield_pred/data/corn_io_il_in.csv')
cropdat = cropdat.dropna()
cropdat['county_ansi'] = cropdat.county_ansi.astype(int)
cropdat['year'] = cropdat.year.astype(int)
cropdat['county_ansi'] = cropdat.county_ansi.astype(str).str.zfill(3)
cropdat['fips'] = cropdat.state_ansi.map(str) + cropdat.county_ansi.map(str)
cropdat = cropdat.drop(['state_ansi', 'county_ansi'], 1)

cropdat = cropdat.pivot_table(index = ['year', 'fips', 'state'], columns = 'data_item', values = 'value')
cropdat = cropdat.reset_index()
cropdat.columns = ['year', 'fips', 'state', 'corn_acres', 'corn_yield']
cropdat['fips'] = cropdat['fips'].astype(str)
cropdat['year'] = cropdat['year'].astype(str)

# Merge data
#cropdat = cropdat.merge(dd, how = 'left', on = ['fips', 'year'])
#cropdat = cropdat.dropna()


# Merge Wolfram data
cropdat = cropdat.merge(dday, how = 'left', on = ['fips', 'year'])

# Trend
cropdat['trend'] = cropdat['year'].astype(int) - 1979
cropdat['trend_sq'] = cropdat['trend']**2

# State Trends
#state_trend = pd.get_dummies(cropdat.state)
#cropdat = pd.concat([cropdat, state_trend], axis=1)
#cropdat['IOWA_trend'] = cropdat['IOWA']*cropdat['trend']
#cropdat['INDIANA_trend'] = cropdat['INDIANA']*cropdat['trend']
#cropdat['ILLINOIS_trend'] = cropdat['ILLINOIS']*cropdat['trend']

#cropdat['IOWA_trend_sq'] = cropdat['IOWA']*cropdat['trend_sq']
#cropdat['INDIANA_trend_sq'] = cropdat['INDIANA']*cropdat['trend_sq']
#cropdat['ILLINOIS_trend_sq'] = cropdat['ILLINOIS']*cropdat['trend_sq']

# Quad precipitation
cropdat['prec_sq'] = cropdat['prec']**2

cropdat['ln_corn_yield'] = np.log(1 + cropdat.corn_yield)

cropdat = cropdat.dropna()
cropdat.to_pickle('/Users/john/Projects/corn_yield_pred/data/full_data.pickle')
cropdat = pd.read_pickle('/Users/john/Projects/corn_yield_pred/data/full_data.pickle')

print(cropdat.head())