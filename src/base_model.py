#imports

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#read data
path = "../data"
train_data = pd.read_json(path + '/train.json/train.json')
##print('train_data shape: ',train_data.shape)
##print(train_data.head())
##print(type(train_data))
##print(train_data.info())

# label and features

print('labels and features')
y = train_data['interest_level']
X = train_data[['bathrooms', 'bedrooms', 'price']]


##X.drop(['interest_level'], axis = 1 ,inplace = True)
##X.drop(['latitude'], axis = 1 ,inplace = True)
##X.drop(['listing_id'], axis = 1 ,inplace = True)
##X.drop(['longitude'], axis = 1 ,inplace = True)
##X.drop(['manager_id'], axis = 1 ,inplace = True)
##X.drop(['photos'], axis = 1 ,inplace = True)
##X.drop(['street_address'], axis = 1 ,inplace = True)



print('X shape: ')
print(X.shape)
##print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

print('X_train and X_test size: ')
print(X_train.shape)
print(X_test.shape)

#base model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
pred = lr_model.predict(X_test)
print('accuracy: ')
print(accuracy_score(y_test, pred))




print('ok')
