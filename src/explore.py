import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_json("../data/train.json/train.json")

# barcharts of numeric features - how many occurances for each level?

##intlevel_count = train_df["interest_level"].value_counts()
##plt.figure()
##sns.barplot(intlevel_count.index, intlevel_count.values)
##plt.xlabel("interest level")
##plt.ylabel("count")
##plt.show()
##
##
##bathrooms_count = train_df["bathrooms"].value_counts()
##plt.figure()
##sns.barplot(bathrooms_count.index, bathrooms_count.values)
##plt.xlabel("bathrooms")
##plt.ylabel("count")
##plt.show()
##
##
##bedrooms_count = train_df["bedrooms"].value_counts()
##plt.figure()
##sns.barplot(bedrooms_count.index, bedrooms_count.values)
##plt.xlabel("bedrooms")
##plt.ylabel("count")
##plt.show()

#distribution of the prices

# Generally scatter plot is to check the relationship bw two variables
# here lets consider x axis as the index (just a number)
##plt.scatter(range(train_df.shape[0]),train_df['price'])
##plt.show()

#there are some outliers, lets remove and plot again
#lets select 99 percentile - 1% of the values are above this, so we are removing
ulimit = np.percentile(train_df["price"].values, 99)
print(ulimit)
price_filtered = train_df[train_df["price"] < ulimit]["price"]
plt.hist(price_filtered)
plt.show()

plt.figure()
sns.distplot(price_filtered.values)
plt.xlabel("price")
plt.show()



#hist? >> value vs instances
##plt.hist(train_df["price"])
##plt.show()

#to-do: learn about violinplot































print("processing done")
