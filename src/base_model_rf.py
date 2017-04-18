import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_df = pd.read_json("../data/train.json/train.json")
test_df = pd.read_json("../data/test.json/test.json")


#create features
train_df["num_photos"] = train_df["photos"].apply(len)
train_df["num_features"] = train_df["features"].apply(len)
train_df["num_desc_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
train_df["created"] = pd.to_datetime(train_df["created"])
train_df["year"] = train_df["created"].dt.year
train_df["month"] = train_df["created"].dt.month
train_df["day"] = train_df["created"].dt.day

numeric_features= ["num_photos", "num_features", "num_desc_words", "year", \
                   "month", "day", "bathrooms", "bedrooms", "latitude", "longitude"]

X = train_df[numeric_features]
y = train_df["interest_level"]
X_train, X_test, y_train, y_test = train_test_split(X, y)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print("processing done")


test_df["num_photos"] = test_df["photos"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)
test_df["num_desc_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))
test_df["created"] = pd.to_datetime(test_df["created"])
test_df["year"] = test_df["created"].dt.year
test_df["month"] = test_df["created"].dt.month
test_df["day"] = test_df["created"].dt.day

X_test = test_df[numeric_features]
y_pred = rf_model.predict_proba(X_test)
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


