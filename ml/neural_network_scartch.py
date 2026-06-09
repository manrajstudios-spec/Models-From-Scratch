import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class neural_net:
    def __init__(self,epochs=1000,lr=0.01,neurons_hidden_layer=16):
        self.epochs = epochs
        self.lr = lr
        self.neurons_hidden_layer = neurons_hidden_layer
        self.w1 = None
        self.b2 = None
        self.w2 = None
        self.b2 = None
    
    def fit(self,X_train,y_train):
        w1 = np.random.randn(X_train.shape[1],self.neurons_hidden_layer) * 0.01
        w2 = np.random.randn(self.neurons_hidden_layer,1) * 0.01
        b1 = np.zeros(self.neurons_hidden_layer)
        b2 = np.zeros(1)

        for _ in range(self.epochs):
            z1 = np.dot(X_train,w1) + b1
            a1 = np.maximum(0,z1)
            z2 = np.dot(a1,w2) + b2
            a2 = 1/(1+np.exp(-z2))

            # BackPropagation

            da2 = a2 - y_train # error
            dw2 = 1/len(y_train) * np.dot(a1.T,da2) 
            db2 = 1/len(y_train) * np.sum(da2)
            da1 = np.dot(da2 , w2.T)
            dz1 = da1 * (z1 >0) 
            dw1 = 1/len(y_train) * np.dot(X_train.T , dz1) 
            db1 = 1/len(y_train) * np.sum(dz1)

            w1 = w1 - self.lr * dw1
            w2 = w2 - self.lr * dw2

            b1 = b1 - self.lr * db1
            b2 = b2 - self.lr * db2

        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2

    def predict(self, X_test):
        z1 = np.dot(X_test, self.w1) + self.b1
        a1 = np.maximum(0,z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = 1/(1+np.exp(-z2))
        return (a2 > 0.5).astype(int)


X,y = load_breast_cancer(return_X_y=True)

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_idx,test_idx in split.split(X,y):
    X_train,X_test = X[train_idx],X[test_idx]
    y_train,y_test = y[train_idx],y[test_idx]

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

y_train = y_train.reshape(-1, 1)
nn = neural_net()
nn.fit(X_train,y_train)

y_pred = nn.predict(X_test)
print(accuracy_score(y_pred,y_test))