import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

class Node:
    def __init__(self,feature=None,split=None,left=None,right=None,value=None):
        # Decision Node

        self.feature = feature
        self.split = split
        self.left = left
        self.right = right

        # Leaf Node
        self.value = value
    

class Decision_Tree_Clasifier:

    def __init__(self, max_depth=2,min_samples_split=2,criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def gini(self,cur_y):
        _,c_single = np.unique(cur_y,return_counts=True)
        p = c_single/c_single.sum()
        return 1 - np.sum(p**2)
    
    def entropy(self,cur_y):
        _,c_single = np.unique(cur_y,return_counts=True)
        p = c_single/c_single.sum()
        return -np.sum(p * np.log2(p))

    def impurity(self,cur_y):
        if self.criterion == 'gini':
            return self.gini(cur_y)
        else:
            return self.entropy(cur_y)

    def best_split(self,cur_X,cur_y):
        ig = -1
        feature = None
        split = None

        for f in range(cur_X.shape[1]):
            splits = cur_X[:,f]

            for s in splits:
                left_X = cur_X[:,f] <= s
                right_X = ~left_X
                left_y = cur_y[left_X]
                right_y = cur_y[right_X]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                weighted_gini = len(left_y)/len(cur_y) * self.impurity(left_y) + len(right_y)/len(cur_y) * self.impurity(right_y)

                cur_ig = self.impurity(cur_y) - weighted_gini

                if cur_ig > ig:
                    ig = cur_ig
                    feature = f
                    split = s
        return feature,split 

    def build_tree(self,cur_X,cur_y,cur_depth=0):
        if cur_depth > self.max_depth or len(cur_y) < self.min_samples_split:
            value = np.bincount(cur_y).argmax()
            return Node(value=value)

        feature,split = self.best_split(cur_X,cur_y)

        left_X = cur_X[:,feature] <= split
        right_X = ~left_X
        left_y = cur_y[left_X]
        right_y = cur_y[right_X]

        cur_left = self.build_tree(cur_X[left_X],left_y,cur_depth=cur_depth+1)
        cur_right = self.build_tree(cur_X[right_X],right_y,cur_depth=cur_depth+1)

        return Node(feature,split,cur_left,cur_right)
    
    def fit(self,X_train,y_train):
        self.root = self.build_tree(X_train,y_train)

    def predict_one(self,node,x):
        if node.value is not None:
            return node.value
        elif x[node.feature] <= node.split:
            return self.predict_one(node.left,x) 
        else:
            return self.predict_one(node.right,x)
    def predict(self,X_test):
        return np.array([self.predict_one(self.root,x) for x in X_test])

        
X, y = load_breast_cancer(return_X_y=True)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

my_tree = Decision_Tree_Clasifier(max_depth=5)
my_tree.fit(X_train, y_train)
y_pred = my_tree.predict(X_test)

sk_tree = DecisionTreeClassifier(max_depth=5)
sk_tree.fit(X_train, y_train)
y_pred_sk = sk_tree.predict(X_test)

print(f"My Model: {classification_report(y_test, y_pred)}")
print(f"Sklearn:  {classification_report(y_test, y_pred_sk)}")