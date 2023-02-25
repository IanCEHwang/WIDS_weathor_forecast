import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import prepro_util
import os

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

import catboost
from catboost import CatBoostRegressor
from catboost import Pool, CatBoostClassifier
import xgboost as xgb




### read train data
train = pd.read_csv('train_data.csv')


### read test data
test = pd.read_csv('test_data.csv')


### preprocessing

### target column
target = 'contest-tmp2m-14d__tmp2m'


### test data
pre_train = prepro_util.preprocess_data(train , 4 , "mean" , target)


### PCA
PCA_pre_train = prepro_util.PCA_transform(pre_train , 0.95 , target)




### split the data
X = pre_train[[col for col in pre_train.columns if col != target]]
y = pre_train[target]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)


## XGBOOST


### train the model - XGBoost
model_xgb = xgb.XGBRegressor(booster = 'gbtree',
                             subsample = 0.8,
                             eta = 0.1, 
                             n_estimaters = 15000,
                             colsample_bytree = 0.4,
                             max_depth = 5,
                             tree_method = 'hist',
                             eval_metric = 'rmse', 
                             objective = 'reg:squarederror')

model_xgb.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_test, y_test)], verbose = 100)



### use RMSE to evaluate
y_pred_xgb = model_xgb.predict(x_test)
mse = mean_squared_error(y_pred_xgb, y_test)

print("XGboost MSE : " ,mse)


try:
    os.makedirs('models')
except:
    pass

### save model
model_xgb.save_model("./models/PCA_95_xgb_md_5.json")


### CAT



### train the model - CatBoost
model_cat = CatBoostRegressor(n_estimators = 15000,
                              eval_metric = 'RMSE',
                              learning_rate = 0.1, 
                              verbose = 1,
                              random_seed = 0).fit(x_train, y_train)

model_cat.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_test, y_test)], verbose = 50)

### save model
model_cat.save_model("./models/PCA_95_cat.json")


### use RMSE to evaluate
y_pred_cat = model_cat.predict(x_test)
mean_squared_error(y_pred_cat, y_test)

print("CATboost MSE : ", mean_squared_error)

