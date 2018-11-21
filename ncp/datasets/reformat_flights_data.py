# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:13:37 2018

@author: GK
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import os


# =============================================================================
# PARAMS
# =============================================================================
#!!!!! Number of features is hardcoded in Tran et al.'s config file.
#!!!!! Manually edit flights_active.py, "config.num_inputs = {}" line for Nfeatures

PATH_TO_FLIGHTS_DATA = './flights.csv' #downloaded data
NP_SAVE_DIR = './'
DATASET_PREFIX = 'flights'
NROWS = 10000
TEST_PCT = .20
SPLIT_MODE = 'sequential'   #'sequential' #'random'
RANDOM_SEED = 12345
usecols = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 
           #'FLIGHT_NUMBER', 'TAIL_NUMBER', 
           #'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 
           'DEPARTURE_DELAY', 
           'CANCELLED']



# =============================================================================
# LOAD
# =============================================================================
np.random.seed(RANDOM_SEED)

df = pd.read_csv(PATH_TO_FLIGHTS_DATA, usecols=usecols, nrows=NROWS, engine = 'python')

#Use only those flights that were not cancelled [have a finite delay time]
df = df.loc[df['CANCELLED']==0]
df.reset_index(drop=True)
#print(df.head(20))
#print(df.tail(20))

#Now remove the cancellation feature which is all 0's for those remaining:
df.drop(columns=['CANCELLED'],inplace=True)


#targets:
y = df['DEPARTURE_DELAY'].values
df.drop(columns=['DEPARTURE_DELAY'],inplace=True)



# =============================================================================
# Modify some features
# =============================================================================

#Make ordinal periodic -> continuous periodic [for day of week]:
dow = df['DAY_OF_WEEK'].values.astype(float)
dow_s = np.sin(2.*np.pi*dow/7.)
dow_c = np.cos(2.*np.pi*dow/7.)
df.drop(columns=['DAY_OF_WEEK'],inplace=True)
df['DAY_OF_WEEK_s'] = dow_s
df['DAY_OF_WEEK_c'] = dow_c

#One hot encode some categoricals:
#Airports [about 1K airports is a lot so ignore for now...]
#Just do not even use tail number...
#Only use airline for now (about 10 airlines)
enc = OneHotEncoder()
airline = enc.fit_transform(df['AIRLINE'].values.reshape(-1, 1))
df.drop(columns=['AIRLINE'],inplace=True)
X = df.values
X = hstack((X,airline))
X=X.toarray()





# =============================================================================
# Convert to format used by Tran et al. github
# =============================================================================
"""
The data set path should be a file prefix for four Numpy files named 
<dataset>-train-inputs.npy, 
<dataset>-train-targets.npy, 
<dataset>-test-inputs.npy, 
<dataset>-test-targets.npy
"""

#Randomly split into train, test sets
#Doing uncorrelated train-test split:
if SPLIT_MODE == 'random':
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_PCT)
#vs. what we actually want to try here with temporally different in-distribution vs. OOD:
elif SPLIT_MODE == 'sequential':
    split_ind = int(TEST_PCT * len(df))
    X_train = X[:split_ind]
    X_test = X[split_ind:]
    y_train = y[:split_ind]
    y_test = y[split_ind:]    

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
#print(X_train, X_test, y_train, y_test)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
np.save(os.path.join(NP_SAVE_DIR,f'{DATASET_PREFIX}-train-inputs.npy'),X_train)
np.save(os.path.join(NP_SAVE_DIR,f'{DATASET_PREFIX}-train-targets.npy'),y_train)
np.save(os.path.join(NP_SAVE_DIR,f'{DATASET_PREFIX}-test-inputs.npy'),X_test)
np.save(os.path.join(NP_SAVE_DIR,f'{DATASET_PREFIX}-test-targets.npy'),y_test)

