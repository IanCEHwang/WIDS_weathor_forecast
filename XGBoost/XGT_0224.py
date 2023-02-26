import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

from catboost import CatBoostRegressor
from catboost import Pool, CatBoostClassifier
import xgboost as xgb

import prepro_util



### read test and train data
train = pd.read_csv('./Desktop/wids/train_data.csv')
#test = pd.read_csv('../../../Desktop/wids/test_data.csv')


### target column
target = 'contest-tmp2m-14d__tmp2m'


### preprocess training data -> rounding lat , lon to 4th + filling na using "mean"
pro_train = prepro_util.preprocess_data(train , 4 , "mean" , target)


### split the data
X = pro_train[[col for col in pro_train.columns if col != target]]
y = pro_train[target]


### train test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)


### train the model - XGBoost
model_xgb = xgb.XGBRegressor(booster = 'gbtree',
                             subsample = 0.8,
                             eta = 0.1, 
                             colsample_bytree = 0.4,
                             max_depth = 5,
                             tree_method = 'hist',
                             eval_metric = 'rmse', 
                             objective = 'reg:squarederror')

model_xgb.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_test, y_test)], verbose = 100)



### use RMSE to evaluate
y_pred_xgb = model_xgb.predict(x_test)
mse = mean_squared_error(y_pred_xgb, y_test)

print("MSE : " ,mse)

### save model
model_xgb.save_model("./Documents/GitHub/WIDS_weathor_forecast/models/model_mean_fill_max_depth_5.json")






