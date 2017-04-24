import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#read train and test datasets
train_df = pd.read_json("../data/train.json/train.json")
test_df = pd.read_json("../data/test.json/test.json")


#create features in train dataset
train_df["num_photos"] = train_df["photos"].apply(len)
train_df["num_features"] = train_df["features"].apply(len)
train_df["num_desc_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
train_df["created"] = pd.to_datetime(train_df["created"])
train_df["year"] = train_df["created"].dt.year
train_df["month"] = train_df["created"].dt.month
train_df["day"] = train_df["created"].dt.day

#list of all numeric features    #to-do consider categorical as well
numeric_features= ["num_photos", "num_features", "num_desc_words", "year", \
                   "month", "day", "bathrooms", "bedrooms", "latitude", "longitude"]
#build train-test features and labels on the train data
X = train_df[numeric_features]
y = train_df["interest_level"]
X_train, X_test, y_train, y_test = train_test_split(X, y)

#build model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("accuracy score of rf model")
print(accuracy_score(y_test, y_pred))
print("processing done")

adaboost_model = AdaBoostClassifier(\
    n_estimators=50, \
    learning_rate=1.0, algorithm='SAMME.R', random_state=None)
adaboost_model.fit(X_train, y_train)
y_pred = adaboost_model.predict(X_test)
print("accuracy score of adaboost_model")
print(accuracy_score(y_test, y_pred))
print("processing done")

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print("accuracy score of lr_model")
print(accuracy_score(y_test, y_pred))
print("processing done")


knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
print("accuracy score of knn_model")
print(accuracy_score(y_test, y_pred))
print("processing done")

#create same features in test dataset that were created in the train dataset

test_df["num_photos"] = test_df["photos"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)
test_df["num_desc_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))
test_df["created"] = pd.to_datetime(test_df["created"])
test_df["year"] = test_df["created"].dt.year
test_df["month"] = test_df["created"].dt.month
test_df["day"] = test_df["created"].dt.day

#create test features with the numeric features

X_test = test_df[numeric_features]
#train rf model on complete train data
##rf_model = RandomForestClassifier()
##rf_model.fit(X, y)
##y_pred = rf_model.predict_proba(X_test)

adaboost_model = AdaBoostClassifier(\
    n_estimators=50, \
    learning_rate=1.0, algorithm='SAMME.R', random_state=None)
adaboost_model.fit(X, y)
y_pred = adaboost_model.predict_proba(X_test)


print("prediction on the test data done")

y_pred_df = pd.DataFrame(y_pred)
y_pred_df.columns = rf_model.classes_
# copying test dataset
test_df_new = test_df
#reseting index in the test dataset as the indexes are not ordered
test_df_new = test_df_new.reset_index(drop=True)

y_pred_df["listing_id"] = test_df_new["listing_id"]

new_cols = ["listing_id", "high", "low", "medium"]
y_pred_new = y_pred_df[new_cols]
y_pred_new.to_csv("submission.csv", index = False)
print("submission.csv is ready")



##y_pred = rf_model.predict_proba(X_test)
##print("prediction on the test data done")
##
##y_pred_df = pd.DataFrame(y_pred)
##y_pred_df.columns = rf_model.classes_
### copying test dataset
##test_df_new = test_df
###reseting index in the test dataset as the indexes are not ordered
##test_df_new = test_df_new.reset_index(drop=True)
##
##y_pred_df["listing_id"] = test_df_new["listing_id"]
##
##new_cols = ["listing_id", "high", "low", "medium"]
##y_pred_new = y_pred_df[new_cols]
##y_pred_new.to_csv("submission.csv", index = False)
##print("submission.csv is ready")

