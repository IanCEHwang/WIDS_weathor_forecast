import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import time
import datetime
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



def pre_process(X):
    ### set index
    X = X.set_index("index")
    
    ### scaler
    scaler = MinMaxScaler()
    
    ### Object encoding
    label_binarizer = LabelBinarizer()
    
    ### handle start date
    X['startdate_int'] = X['startdate'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, "%m/%d/%y").timetuple()))
    ### drop startdate
    X = X.drop('startdate' , axis = 1)
    
    ### encode region
    X['climateregions__climateregion'] = label_binarizer.fit_transform(X['climateregions__climateregion'])
    
    ### fill null
    X = X.fillna(method = "ffill")
    
    X = scaler.fit_transform(X.iloc[0 : X.shape[0],:])
    return X


### read data
train = pd.read_csv("train_data.csv")

### set index
train = train.set_index("index")

### preprocess
x = pre_process(train[[col for col in train.columns if col!='contest-tmp2m-14d__tmp2m']])
y = train['contest-tmp2m-14d__tmp2m']


### split train test
x_train , x_test , y_train , y_test = train_test_split(
        x , y , test_size = 0.3 , random_state = 87
)


### make linear model
model = LinearRegression().fit(x_train , y_train)


### Test sample
model.score(x_test , y_test)


### Predict on OOS data


### read data
test = pd.read_csv("test_data.csv")

### preprocess
test_processed = pre_process(test)

### predict
predictions = model.predict(test_processed)

### create output dataframe
temp_dict = {'index' : list(test['index']) , 'contest-tmp2m-14d__tmp2m' : predictions}
output = pd.DataFrame(temp_dict)
output = output.set_index("index")

### output
output.to_csv("trial1_0218.csv")


