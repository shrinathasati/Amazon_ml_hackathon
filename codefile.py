#importing library
import numpy as np
import pandas as pd
import sklearn
import matplotlib

#read train.csv
data_train=pd.read_csv("train.csv")

#missing value handle
data_train.dropna(inplace=True)

#divide dataset
x_train=data_train.iloc[:,:5]
y_train=data_train.iloc[:,5]

#read test.csv
data_test=pd.read_csv("test.csv")

#read submission.csv
data=pd.read_csv("sample_submission.csv")
y_test=data.iloc[:,1]
data_test['PRODUCT_LENGTH']=y_test


data_test.dropna(inplace=True)
x_test=data_test.iloc[:,:5]


#Mapping str to number using encoding
from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()

#for training dataset encoding
y_train_title=x_train['TITLE']
le.fit(y_train_title)
x_train['TITLE'] = le.transform(y_train_title)


y_train_description= x_train['DESCRIPTION']
le.fit(y_train_description)
x_train['DESCRIPTION']=le.transform(y_train_description)


y_train_BULLET_POINTS= x_train['BULLET_POINTS']
le.fit(y_train_BULLET_POINTS)
x_train['BULLET_POINTS']=le.transform(y_train_BULLET_POINTS)


#for testing dataset encoding
y_test_title=x_test['TITLE']
le.fit(y_test_title)
x_test['TITLE'] = le.transform(y_test_title)

y_test_description= x_test['DESCRIPTION']
le.fit(y_test_description)
x_test['DESCRIPTION']=le.transform(y_test_description)

y_test_BULLET_POINTS= x_test['BULLET_POINTS']
le.fit(y_test_BULLET_POINTS)
x_test['BULLET_POINTS']=le.transform(y_test_BULLET_POINTS) 


#model training
from sklearn import linear_model
model=linear_model.LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)



#checking for shape
print(y_predict.shape)
print(data_test['PRODUCT_LENGTH'].shape)


# finding the accuracy
from sklearn import metrics
score = max( 0 , 100*(1-metrics.mean_absolute_percentage_error(data_test['PRODUCT_LENGTH'],y_predict)))
print(score)


#saving file in csv format
df=pd.DataFrame(data)
df.to_csv("output.csv",index=False)