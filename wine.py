from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

wine = load_wine()
X = pd.DataFrame(wine.data, columns = wine.feature_names)
y = pd.Series(wine.target)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=1)

dtclf = DecisionTreeClassifier(max_depth = 1, criterion = 'gini', random_state = 1)
dtclf.fit (X_train, y_train)

from sklearn.metrics import accuracy_score


dtclf_train_sc = accuracy_score(y_train, dtclf.predict(X_train))
dtclf_test_sc = accuracy_score(y_test, dtclf.predict(X_test))
print('Decision tree train/test accuracies %.3f/%.3f' % (dtclf_train_sc, dtclf_test_sc))