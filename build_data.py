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




cropdat = pd.read_csv('/Users/john/Projects/corn_yield_pred/data/corn_io_il_in.csv')
cropdat = cropdat.dropna()
cropdat['county_ansi'] = cropdat.county_ansi.astype(int)
cropdat['year'] = cropdat.year.astype(int)
cropdat['county_ansi'] = cropdat.county_ansi.astype(str).str.zfill(3)
cropdat['fips'] = cropdat.state_ansi.map(str) + cropdat.county_ansi.map(str)
cropdat = cropdat.drop(['state_ansi', 'county_ansi', 'state'], 1)

cropdat = cropdat.pivot_table(index = ['year', 'fips'], columns = 'data_item', values = 'value')
cropdat = cropdat.reset_index()
cropdat.columns = ['year', 'fips', 'corn_acres', 'corn_yield']
cropdat['fips'] = cropdat['fips'].astype(str)
cropdat['year'] = cropdat['year'].astype(str)

cropdat = cropdat.merge(dd, how = 'left', on = ['fips', 'year'])
cropdat = cropdat.dropna()

# Trend
cropdat['trend'] = cropdat['year'].astype(int) - 1979
cropdat['trend_sq'] = cropdat['trend']**2

cropdat.to_pickle('/Users/john/Projects/corn_yield_pred/data/full_data.pickle')
cropdat = pd.read_pickle('/Users/john/Projects/corn_yield_pred/data/full_data.pickle')