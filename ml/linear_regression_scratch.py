# %%

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class linear_regression:

    def __init__(self,learning_rate,ephochs):
        self.lr = learning_rate
        self.ephochs = ephochs
        self.w = None
        self.b = None
        self.loss_hist = []

    def fit(self,X_train,y_train):
        w = np.zeros(X_train.shape[1])
        b = 0
        
        for _ in range(self.ephochs):
            y_pred = np.dot(X_train,w) + b
            
            dw = 2/len(X_train) * X_train.T @ (y_pred - y_train)
            db = 2/len(X_train) * np.sum(y_pred - y_train)

            w = w - self.lr * dw
            b = b - self.lr * db

        self.w = w
        self.b = b
        loss = np.mean((y_pred - y_train) **2)
        self.loss_hist.append(loss)

    def predict(self,X_test):
        return np.dot(X_test,self.w) + self.b
    


X,y = fetch_california_housing(return_X_y=True)

X = (X - X.mean(axis=0)) / X.std(axis=0)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

reg_scratch = linear_regression(0.01,1000)

reg_scratch.fit(X_train,y_train)

y_pred_scratch = reg_scratch.predict(X_test)

reg = LinearRegression()

reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

print(f"My Modle {mean_squared_error(y_pred_scratch,y_test)}")
print(f"Sk Learn: {mean_squared_error(y_pred,y_test)}")
