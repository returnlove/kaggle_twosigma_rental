import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_json("../data/train.json/train.json")

# barcharts of numeric features - how many occurances for each level?

intlevel_count = train_df["interest_level"].value_counts()
plt.figure()
sns.barplot(intlevel_count.index, intlevel_count.values)
plt.xlabel("interest level")
plt.ylabel("count")
plt.show()


bathrooms_count = train_df["bathrooms"].value_counts()
plt.figure()
sns.barplot(bathrooms_count.index, bathrooms_count.values)
plt.xlabel("bathrooms")
plt.ylabel("count")
plt.show()


bedrooms_count = train_df["bedrooms"].value_counts()
plt.figure()
sns.barplot(bedrooms_count.index, bedrooms_count.values)
plt.xlabel("bedrooms")
plt.ylabel("count")
plt.show()




print("processing done")
