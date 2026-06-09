import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,root_mean_squared_error

class logistic_regression:
    
    def __init__(self,epochs,learning_rate):
        self.epochs = epochs
        self.lr = learning_rate
        self.w = None
        self.b = 0

    def fit(self,X_train,y_train):
        w = np.zeros(X_train.shape[1])
        b = 0 

        for i in range(self.epochs):
            y_pred = 1 / (1 + np.exp(-np.dot(X_train, w) - b))

            dw = 1/len(X_train) * X_train.T @ (y_pred - y_train)
            db = 1/len(X_train) * np.sum(y_pred - y_train)
            
            w = w - self.lr * dw
            b = b - self.lr * db

        self.w = w
        self.b = b

    def predict(self, X_test):
        proba = 1 / (1 + np.exp(-np.dot(X_test, self.w) - self.b))
        return (proba >= 0.5).astype(int)
    

X,y = load_breast_cancer(return_X_y=True)

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_idx , test_idx in split.split(X,y):
    X_train,X_test = X[train_idx],X[test_idx]
    y_train,y_test = y[train_idx],y[test_idx]

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

clf_scratch = logistic_regression(5000,0.1)
clf_scratch.fit(X_train,y_train)
y_pred_scratch = clf_scratch.predict(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print(f"My Modle {accuracy_score(y_pred_scratch,y_test)}")
print(f"Sk Learn Modle {accuracy_score(y_pred,y_test)}")
