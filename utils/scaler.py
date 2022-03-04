import numpy as np

class MyStandardScaler():
    def __init__(self):
        pass
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        
    def transform(self, X):
        X_tr = np.copy(X)
        X_tr -= self.mean_
        X_tr /= self.std_
        return X_tr

class MyMinMaxScaler():
    def __init__(self):
        pass
    
    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        
    def transform(self, X):
        X_tr = np.copy(X)
        X_tr -= self.min_
        X_tr /= (self.max_ - self.min_)
        return X_tr