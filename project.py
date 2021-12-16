# A machine learning library used for linear regression
from requests.adapters import Response
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
# numpy and pandas will be used for data manipulation
import numpy as np
import pandas as pd
# matplotlib will be used for visually representing our data
import matplotlib.pyplot as plt
# Quandl will be used for importing historical oil prices
import quandl
from pymongo import MongoClient
import json, requests
import pickle

URI = f"mongodb+srv://TaesukKim:taesuk123@cluster0.iwxmo.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
client = MongoClient({URI})
db = client["myFirstDatabase"]
myCollection = db["silver"]
# Setting our API key
quandl.ApiConfig.api_key = "ir5mHycopSzP__urB69Y"

# Importing our data
data = quandl.get("LBMA/SILVER")
df = quandl.get("LBMA/SILVER",returns="pandas")
records = json.loads(df.T.to_json()).values()
myCollection.insert_many(records)

data.head()

# Setting the text on the Y-axis
plt.ylabel("Silver")

# Setting the size of our graph
data.USD.plot(figsize=(10,5))

data['av3'] = data['USD'].shift(1).rolling(window=3).mean()
data['av6']= data['USD'].shift(1).rolling(window=6).mean()


# Dropping the NaN values
data = data.dropna()

# Initialising X and assigning the two feature variables
X = data[['av3','av6']]

# Getting the head of the data
X.head()

# Setting-up the dependent variable
y = data['USD']

# Getting the head of the data
y.head()


# Setting the training set to 80% of the data
training = 0.8
t = int(training*len(data))

# Training dataset
X_train = X[:t]
y_train = y[:t]

# Testing dataset
X_test = X[t:]
y_test = y[t:]


# Generate the coefficient and constant for the regression
model = LinearRegression().fit(X_train,y_train)

predicted_price = model.predict(X_test)
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])
predicted_price.plot(figsize=(10,5))
y_test.plot()
plt.legend(['Predicted Price','Actual Price'])
plt.ylabel("Silver")
plt.show()

# Computing the accuracy of our model
R_squared_score = model.score(X[t:],y[t:])*100
accuracy = ("{0:.2f}".format(R_squared_score))
print ("The model has a " + accuracy + "% accuracy.")

with open('silver.pkl', 'wb') as m:
    pickle.dump(model,m)