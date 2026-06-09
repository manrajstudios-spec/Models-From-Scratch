import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_one(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        return np.bincount(k_labels).argmax()

    def predict(self, X_test):
        return np.array([self.predict_one(x) for x in X_test])

X, y = load_breast_cancer(return_X_y=True)
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

my_knn = KNN(k=5)
my_knn.fit(X_train, y_train)
y_pred = my_knn.predict(X_test)

sk_knn = KNeighborsClassifier(n_neighbors=5)
sk_knn.fit(X_train, y_train)
y_pred_sk = sk_knn.predict(X_test)

print(f"My Model: {accuracy_score(y_test, y_pred):.4f}")
print(f"Sklearn:  {accuracy_score(y_test, y_pred_sk):.4f}")