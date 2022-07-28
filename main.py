import pandas as pd

# --------------------------------------------------------------#
# Model Training                                               #
# --------------------------------------------------------------#

# Navie Bayes
def model_test(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import MultinomialNB

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    print(mnb.score(X_test, y_test)*100,"% Accurate")

# Data Selection
####### --------------------------------------- #######
dataset = pd.read_csv("dataset/dataset.csv")

# print(dataset.head(10))

# Data Preprocessing
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
dataset['name'] = labelencoder.fit_transform(dataset['name'])
dataset['gender'] = labelencoder.fit_transform(dataset['gender'])
dataset['weekday'] = labelencoder.fit_transform(dataset['weekday'])

# print(dataset.head(10))

# Data Speration
X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, -1].values

# Training Data And testing Data Speration
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Training Data and Testing Data using Stratified Kflod

from sklearn.model_selection import StratifiedKFold

sskf = StratifiedKFold(n_splits=10)
for train_index, test_index in sskf.split(X=X, y=y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index] \
        , y[train_index], y[test_index]
    model_test(X_train,X_test, y_train, y_test)





# Model testing
# y_pred=mnb.predict(X_test)

# Model Acurracy
from sklearn.metrics import classification_report

# cc = classification_report(y_test, y_pred=y_pred)
