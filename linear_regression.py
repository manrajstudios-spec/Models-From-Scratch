import numpy as np 
from sklearn.datasets import fetch_california_housing

X,y = fetch_california_housing(return_X_y=True)


X = (X-X.mean(axis=0))/X.std(axis=0)    
w = np.zeros(X.shape[1])
b = 0

def predict(w,b):
    return np.dot(X,w) + b

for i in range(1000):
    loss = (1/len(X)) * np.sum((predict(w,b) - y)**2)
    print(f"Epoch {i} | Loss: {loss:.4f}")

    dw = 2/len(X) * X.T @ (predict(w,b) - y)
    db = 2/len(X) * np.sum(predict(w,b) - y) 

    w = w - 0.01 * dw
    b = b - 0.01 * db

    loss = (1/len(X)) * np.sum((predict(w,b) - y)**2)
    print(f"Epoch {i} | Loss: {loss:.4f}")

print(w,b)