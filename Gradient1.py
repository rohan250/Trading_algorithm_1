'''
#  In this program, I will use Gradient Boosting to generate a predictive series of data on stock prices
# I will run the stock data through the AX, ATR, RSI, MFI programs i wrote earlier and store al the data in one csv
# Then, I will load the csv into the program and use them in the feature matrix
# My target variable in this case is the daily returns which I define here as the percent difference
# between the closing price and the previous closing price

'''
# Step1: Importing the basicc libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.metrics import mean_squared_error as MSE

df = pd.read_csv("amzn3.csv")
# Dropping all NaN values
df2 = df.dropna()
#print (df2.head(5))
# Since I am working with a time series, I will split the data into the first 70% for training and last 30% for testing
new_df = df2['Date']
train_size = round (0.7*len(df2))
df_train = df2.iloc [:train_size,:]
df_test = df2.iloc [train_size:,:]

# The dataframe in this instance contains columns with Open, High, Low, Close, Volume, and 4 momentum indicators
# My feature variables in this case are only the indicators
# My target variable is the closing price

# 'Close', 'Open', 'High', 'Low', 'Volume',
# 'Close', 'Open', 'High', 'Low', 'Volume',

X_train = df_train.drop (columns = ['Close', 'Date'])
X_test = df_test.drop (columns = ['Close', 'Date'])
Date_test = df_test['Date']
y_train = df_train['Close']
y_test = df_test ['Close']

from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.ensemble import GradientBoostingClassifier as gbc

gb = gbr (n_estimators = 400, max_depth = 2, learning_rate = 0.1 )
gb.fit(X_train, y_train)
y_pred = gb.predict (X_test)

difference = y_test - y_pred
PCTdiff = (difference/y_test) * 100

import matplotlib.pyplot as plt
plt.style.use("seaborn-dark-palette")

figure, axis = plt.subplots()
axis.plot (Date_test, PCTdiff)
#axis.plot(y_test)
plt.show()



