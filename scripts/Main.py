import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold,StratifiedKFold,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin,ClassifierMixin
from Model import MLPClassifier,XGBClassifier,LogRegClassifier
from Data import Preprocessing
import pickle
def model():
    data=pd.read_csv('./bank-additional-full.csv',sep=';')
    reduced_data=Preprocessing().fit_transform(data)
    smote = SMOTE(sampling_strategy='minority')
    X_sm, y_sm = smote.fit_resample(reduced_data.drop(['y'],axis=1), reduced_data.y)
    Xf_train,Xf_test,yf_train,yf_test=train_test_split(X_sm,y_sm,test_size=0.1,random_state=1234,shuffle=True)
    model_mlp=MultiLayerClassifier()
    model_xgb=XGBClassifier()
    model_logreg=LogRegClassifier()
    model_mlp.fit(Xf_train,yf_train)
    model_xgb.fit(Xf_train,yf_train)
    model_logreg.fit(Xf_train,yf_train)
    print ("The accuracy(f1 score) for {0} mode is {1}".format('Multilayer perceptron',model_mlp.score(Xf_test,yf_test)))
    print ("The accuracy(f1 score) for {0} mode is {1}".format('XGBClassifier',model_xgb.score(Xf_test,yf_test)))
    print ("The accuracy(f1 score) for {0} mode is {1}".format('Logistic Regression',model_logreg.score(Xf_test,yf_test)))
model()
