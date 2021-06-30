import pandas as pd
import numpy as np
# Scikit functions
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier as KNC




def learning(X_train, y_train, alg_name):
    """
    This function features a ML pipeline that trains the data based on a algorithm given as input (str).
    Moreover, this pipeline includues a One Hot Encoder in order to deal with classified data
    """
    pipe = Pipeline([
        ('encoder', OneHotEncoder(sparse=False, handle_unknown='ignore')),
        #('simp', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),   #I am doubtful about this one
        ('scaler', StandardScaler()),
        ('classifier', alg_name)
    ])
    result = pipe.fit(X_train, y_train)
    return result


def scoring(X_test, y_test, pipe_fit):
    print('Testing score for the RFC: ', pipe_fit.score(X_test, y_test))