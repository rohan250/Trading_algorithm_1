'''
Following: https://towardsdatascience.com/under-the-hood-of-gradient-boosting-and-its-python-implementation-99cc63efd24d
Gradient Boosting:
- a great alternative to ADABoost
- uses the gradient descent algorithm and tries to minimize the errors (residuals) through gradient descent
- each new tree is added to the ensemble by correcting the errors of previous trees
- each new tree is fitted ON THE RESIDUALS made by previous trees' predictions

'''

# Residual = actual value - predicted value
# - can be positive or negative

# Importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split as TTS
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.metrics import mean_squared_error as MSE

df = pd.read_csv('cali_housing.csv')
print (df.head(10))

# target variable: MedHouseVal
X = df.drop(columns = 'MedHouseVal') # The dataframe without the target variable is the feature matrix (contains all regressors)
y = df['MedHouseVal']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = TTS (X,y, test_size=0.20, random_state=1)

# 1st tree in the ensemble
tree_1  = dtr(max_depth=1, random_state=1)
tree_1.fit(X_train, y_train)

# Making the predictions of the first tree
tree_1_pred = tree_1.predict(X_train)
tree_1_res = y_train - tree_1_pred


# Training the second tree on the residuals of the first tree
# 2nd tree in the ensemble
tree_2 = dtr(max_depth=1, random_state=1)
tree_2.fit(X_train, tree_1_res)
tree_2_pred = tree_2.predict(X_train)
tree_2_res = tree_1_res - tree_2_pred


# 3rd tree in the ensemble
tree_3 = dtr(max_depth=1, random_state=1)
tree_3.fit(X_train, tree_2_res)
tree_3_pred = tree_3.predict (X_train)
tree_3_res = tree_2_res - tree_3_pred

'''
This is the end of the third iteration.
We can continue to build trees until the residuals approach 0.
There can be millions of iterations (defined by *n_estimators*)
'''

# Now, we calcualte the RMSE value:
y1_pred = tree_1.predict (X_test)
y2_pred = tree_2.predict(X_test)
y3_pred = tree_3.predict(X_test)

y_pred = y1_pred + y2_pred + y3_pred
RMSE = MSE(y_test, y_pred)**0.5
print (RMSE)

# The RMSE has been found to be approximately 0.86 in y-units
# This value can be minimized by increasing the number of iterations, but it is not practical to do this manually many times
# Therefore, we use the SciKitLearn GradientBoostingRegressor()-for regression-
# and GradientBoostingClassifier()--for classification--to easily implement gradient boosting


'''
Using Scikitlearn's gradient boosting regressor to train a gradient boosting model

'''
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import GradientBoostingClassifier as GBC

gb = GBR (n_estimators = 100, max_depth=1, learning_rate= 1.0)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
RMSE = MSE (y_test, y_pred) ** 0.5
print (RMSE)
# Measuring the effect of n_estimators

x = list(range(1, 520, 20))
y = []

for i in x:
  gb = GBR(n_estimators=i,max_depth=1, learning_rate=1.0)

  gb.fit(X_train, y_train)
  y_pred = gb.predict(X_test)

  RMSE= MSE(y_test, y_pred)**0.5
  y.append(RMSE)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.title("Effect of n_estimators", pad=20)
plt.xlabel("Number estimators")
plt.ylabel("Test RMSE of Gradient Boosting")
plt.plot(x,y)
plt.show()

# Measuring the effect of learning_rate
x = [1, 2, 3, 4, 5]
y = []

for i in x:
  gb = GBR(n_estimators=300,
                                 max_depth=i,
                                 learning_rate=1.0)

  gb.fit(X_train, y_train)
  y_pred = gb.predict(X_test)

  RMSE= MSE(y_test, y_pred)**0.5
  y.append(RMSE)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.title("Effect of max_depth", pad=20)
plt.xlabel("Max depth")
plt.ylabel("Test RMSE of Gradient Boosting")
plt.plot(x, y)
plt.show()


# Measuring the effect of learning_rate
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
y = []

for i in x:
  gb = GBR(n_estimators=300,
                                 max_depth=2,
                                 learning_rate=i)

  gb.fit(X_train, y_train)
  y_pred = gb.predict(X_test)

  RMSE= MSE(y_test, y_pred)**0.5
  y.append(RMSE)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.title("Effect of learning_rate", pad=20)
plt.xlabel("Learning rate")
plt.ylabel("Test RMSE of Gradient Boosting")
plt.plot(x, y)
plt.show()


hyperparameter_space = {'n_estimators':[300, 350, 400, 450, 500],
                        'learning_rate':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'max_depth':[1, 2]}

#Finding the optimal parameters using Randomised search

from sklearn.model_selection import RandomizedSearchCV as RSCV

rs = RSCV(GBR(),
                        param_distributions=hyperparameter_space,
                        n_iter=10, scoring="neg_root_mean_squared_error",
                        random_state=1, n_jobs=-1, cv=5)


rs.fit(X_train, y_train)
print("Optimal hyperparameter combination:", rs.best_params_)
