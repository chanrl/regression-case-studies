import pandas as pd
import numpy as np
from typing import Sequence

from sklearn.ensemble import RandomForestRegressor
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, mean_squared_error, r2_score

from rfpimp import *


def test(X, y, n_estimators=50,
         max_features='auto', min_samples_leaf=1):
    rf = RandomForestRegressor(n_estimators=n_estimators,
                               n_jobs=-1,
                               oob_score=True,
                               max_features=max_features, 
                               min_samples_leaf=min_samples_leaf)
    rf.fit(X, y)
    oob = rf.oob_score_
    n = rfnnodes(rf)
    h = np.median(rfmaxdepths(rf))
    print(f"OOB R^2 {oob:.5f} using {n:,d} tree nodes with {h} median tree height")
    return rf, oob
        
def df_split_dates(df,colname):
    df["saleyear"] = df[colname].dt.year
    df["salemonth"] = df[colname].dt.month
    df["saleday"] = df[colname].dt.day
    df["saledayofweek"] = df[colname].dt.dayofweek
    df["saledayofyear"] = df[colname].dt.dayofyear
    df[colname] = df[colname].astype(np.int64) # convert to seconds since 1970
    # age can be nan since YearMade can be nan
    df['age'] = df['saleyear'] - df['YearMade'] # synthesize age

def extract_sizes(df, colname):
    df[colname] = df[colname].str.extract(r'([0-9.]*)', expand=True)
    df[colname] = df[colname].replace('', np.nan)
    df[colname] = pd.to_numeric(df[colname])
    
def df_normalize_strings(df):
    for col in df.columns:
        if is_string_dtype(df[col]) or is_object_dtype(df[col]):
            df[col] = df[col].str.lower()
            df[col] = df[col].fillna(np.nan) # make None -> np.nan
            df[col] = df[col].replace('none or unspecified', np.nan)
            df[col] = df[col].replace('none', np.nan)
            df[col] = df[col].replace('#name?', np.nan)
            df[col] = df[col].replace('', np.nan)

def df_cat_to_catcode(df:pd.DataFrame):
    for colname in df.columns:
        if is_categorical_dtype(df[colname]):
            df[colname] = df[colname].cat.codes + 1
            
def fix_missing_num(df, colname):
    df[colname+'_na'] = pd.isnull(df[colname])
    df[colname].fillna(df[colname].median(), inplace=True)
    
def clean(df):
    del df['MachineID'] # dataset has inconsistencies
    del df['SalesID']   # unique sales ID so not generalizer

    df['auctioneerID'] = df['auctioneerID'].astype(str)

    df_normalize_strings(df)

    extract_sizes(df, 'Tire_Size')
    extract_sizes(df, 'Undercarriage_Pad_Width')

    df.loc[df['YearMade']<1950, 'YearMade'] = np.nan
    df.loc[df.eval("saledate.dt.year < YearMade"), 'YearMade'] =         df['saledate'].dt.year

    df.loc[df.eval("MachineHoursCurrentMeter==0"),
           'MachineHoursCurrentMeter'] = np.nan
    
def df_order_product_size(df):
    sizes = {np.nan:0, 'mini':1, 'compact':1, 'small':2, 'medium':3,
             'large / medium':4, 'large':5}
    df['ProductSize'] = df['ProductSize'].map(sizes).values

def onehot(df, colname):
    ascat = df[colname].astype('category').cat.as_ordered()
    onehot = pd.get_dummies(df[colname], prefix=colname, dtype=bool)
    del df[colname]
    df = pd.concat([df, onehot], axis=1)
    # return altered dataframe and column training categories
    return df, ascat.cat.categories

def split_fiProductClassDesc(df):
    df_split = df.fiProductClassDesc.str.split(' - ',expand=True).values
    df['fiProductClassDesc'] = df_split[:,0] 
    df['fiProductClassSpec'] = df_split[:,1] # temporary column
    pattern = r'([0-9.\+]*)(?: to ([0-9.\+]*)|\+) ([a-zA-Z ]*)'
    spec = df['fiProductClassSpec']
    df_split = spec.str.extract(pattern, expand=True).values
    df['fiProductClassSpec_lower'] = pd.to_numeric(df_split[:,0])
    df['fiProductClassSpec_upper'] = pd.to_numeric(df_split[:,1])
    df['fiProductClassSpec_units'] = df_split[:,2]
    del df['fiProductClassSpec'] # remove temporary column
    
def feature_eng(X): # for later use
    df_split_dates(X, 'saledate')
    df_order_product_size(X)
    split_fiProductClassDesc(X)

    X, hf_cats = onehot(X, 'Hydraulics_Flow')
    # normalize categories first then one-hot encode
    X['Enclosure'] = X['Enclosure'].replace('erops w ac', 'erops ac')
    X['Enclosure'] = X['Enclosure'].replace('no rops', np.nan)
    X, enc_cats = onehot(X, 'Enclosure')
    catencoders = {'Hydraulics_Flow':hf_cats,
                   'Enclosure':enc_cats}

    return X, catencoders

def df_fix_missing_nums(df:pd.DataFrame) -> dict:
    medians = {}  # column name to median
    for colname in df.columns:
        if is_numeric_dtype(df[colname]):
            medians[colname] = df[colname].median(skipna=True)
            fix_missing_num(df, colname)
    return medians

def df_string_to_cat(df:pd.DataFrame) -> dict:
    catencoders = {}
    for colname in df.columns:
        if is_string_dtype(df[colname]) or is_object_dtype(df[colname]):
            df[colname] = df[colname].astype('category').cat.as_ordered()
            catencoders[colname] = df[colname].cat.categories
    return catencoders

def numericalize(X, catencoders):
    medians = df_fix_missing_nums(X)            
    e = df_string_to_cat(X)
    catencoders.update(e)
    df_cat_to_catcode(X)    
    return medians

def df_fix_missing_test_nums(df_test, medians):
    for colname in medians:
        df_test[colname+'_na'] = pd.isnull(df_test[colname])
        df_test[colname].fillna(medians[colname], inplace=True)
        
def df_apply_cats(df_test:pd.DataFrame, catencoders:dict):
    for colname,encoder in catencoders.items():
        # encode with categories from training set
        df_test[colname] =             pd.Categorical(df_test[colname],
                           categories=encoder, ordered=True)
        
def onehot_apply_cats(df_test, colname, catencoders):
    df_test[colname] =         pd.Categorical(df_test[colname],
                       categories=catencoders[colname],
                       ordered=True)
    onehot = pd.get_dummies(df_test[colname], prefix=colname, dtype=bool)
    del df_test[colname]
    df_test = pd.concat([df_test, onehot], axis=1)
    del catencoders[colname] # simplify df_apply_cats()
    return df_test

def feature_eng_test(df_test, catencoders):
    df_split_dates(df_test, 'saledate')
    df_order_product_size(df_test)
    split_fiProductClassDesc(df_test)

    df_test = onehot_apply_cats(df_test, 'Hydraulics_Flow', catencoders)
    df_test['Enclosure'] = df_test['Enclosure'].replace('erops w ac', 'erops ac')
    df_test['Enclosure'] = df_test['Enclosure'].replace('no rops', np.nan)
    df_test = onehot_apply_cats(df_test, 'Enclosure', catencoders)

    return df_test

def numericalize_test(df_test:pd.DataFrame, medians:dict, catencoders:dict):
    df_apply_cats(df_test, catencoders)
    df_fix_missing_test_nums(df_test, medians)
    df_cat_to_catcode(df_test)
    
def test_valid(X, y, X_valid, y_valid, n_estimators=200,
               max_features='auto', min_samples_leaf=1):
    X_valid = X_valid.reindex(columns=X.columns)
    rf = RandomForestRegressor(n_estimators=n_estimators,
                               n_jobs=-1,
                               oob_score=True,
                               max_features=max_features, 
                               min_samples_leaf=min_samples_leaf)
    rf.fit(X, y)
    n = rfnnodes(rf)
    h = np.median(rfmaxdepths(rf))
    y_pred = rf.predict(X_valid)
    mae_valid = mean_absolute_error(np.exp(y_valid), np.exp(y_pred))
    rmsle_valid = np.sqrt( mean_squared_error(y_valid, y_pred) )
    r2_score_valid = rf.score(X_valid, y_valid)
    print(f"OOB R^2 {rf.oob_score_:.5f} using {n:,d} tree nodes {h} median tree height")
    print(f"Validation R^2 {r2_score_valid:.5f}, RMSLE {rmsle_valid:.5f}, MAE ${mae_valid:.0f}")
    return rf, r2_score_valid, rmsle_valid, mae_valid

def select_features(X, y, X_valid, y_valid, drop=0.10):
   min_rmsle = 99999
   X_valid = X_valid.reindex(columns=X.columns)
   rf, _, rmsle, _ = test_valid(X, y, X_valid, y_valid,
                                max_features=.3, min_samples_leaf=2)
   I = importances(rf, X_valid, y_valid)
   features = list(I.index)
   keep = best_features = features
   n = int(.9/drop) # how many iterations? get to 90%
   for i in range(1,n+1):
       X2 = X[keep]
       X_valid2 = X_valid[keep]
       print(f"Num features = {len(keep)}")
       rf2, _, rmsle, _ = test_valid(X2, y, X_valid2, y_valid,
                                     max_features=.3, min_samples_leaf=2)
       if rmsle < min_rmsle:
           min_rmsle = rmsle
           best_features = keep
       I2 = importances(rf2, X_valid2, y_valid) # recompute since collinear
       features = list(I2.index)
       keep = features[0:int(len(features)*(1-drop))]

   return min_rmsle, best_features