class XGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        x_train_total, y_train_total = X,y
        dtrain = xgb.DMatrix(x_train_total, y_train_total)
        # specify parameters via map
        params = {"objective": "binary:logistic", # for classification
          "booster" : "gbtree",   # use tree based models 
          "eta": 0.01,   # learning rate
          "max_depth": 10,    # maximum depth of a tree
          "subsample": 1.0,    # Subsample ratio of the training instances
          "colsample_bytree": 0.7,   # Subsample ratio of columns when constructing each tree
          "silent": 1,   # silent mode
          "seed": 10 ,  # Random number seed
           "n_estimators":300,
          }
        num_round = 2000
        self.model = xgb.train(params, dtrain, num_round)
        return self
    
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        predict=np.expm1(0.995*(self.model.predict(dtest)))
        return predict
    def metric(self,y, t, threshold=0.5):
        try:
            t = t.get_label()
        except AttributeError:
            pass
        y_bin = (y > threshold).astype(int) # works for both type(y) == <class 'numpy.ndarray'> and type(y) == <class 'pandas.core.series.Series'>
        return 'f1',f1_score(t,y_bin)
    def score(self,X,y):
        y_pred=self.predict(X)
        score=self.metric(y_pred,y)
        return score
        
class LogRegClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        stratfold=StratifiedKFold(n_splits=5)
        # Create logistic regression
        logistic = LogisticRegression()
        # Create regularization penalty space
        penalty = ['l1', 'l2']
        # Create regularization hyperparameter space
        C = np.logspace(0, 4, 100)

        # Create hyperparameter options
        hyperparameters = dict(C=C, penalty=penalty)
        clf = GridSearchCV(logistic, hyperparameters, cv=stratfold, verbose=0,scoring='f1')
        self.model = clf.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        predict=self.model.predict(X)
        return predict
    def score(self,X,y):
        score=self.model.score(X,y)
        return score
        
class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        stratfold=StratifiedKFold(n_splits=5)
        parameters = {'solver': ['lbfgs'], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 
                      'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 
                      'random_state':[0,1,2,3,4,5,6,7,8,9]}
        clf = RandomizedSearchCV(MLPClassifier(), parameters, cv=stratfold,n_jobs=-1,scoring='f1')
        self.model = clf.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        predict=self.model.predict(X)
        return predict
    def score(self,X,y):
        score=self.model.score(X,y)
        return score
