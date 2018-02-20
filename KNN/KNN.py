'''
Input:  X_train: (M, N) matrix
        y_train: (M, ) vector
        X_test: (K, L) matrix
        y_test: (K, ) vector
''' 
import numpy as np
import numpy.linalg as la

class KNN():
    def __init__(self, k):
        self.k = k 

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict_(self, one_data):   
        dist = la.norm(self.X_train - one_data, ord=2, axis=1)
        index = dist.argsort()
        class_count = {}
        for i in range(self.k):
            vote_class = self.y_train[index[i]]
            class_count[vote_class] = class_count.get(vote_class, 0) + 1

        sorted_class_count = sorted(class_count.items(), key=lambda d: d[1], reverse=True)

        return sorted_class_count[0][0]

    def predict(self, X_test):
        return np.array([self.predict_(val) for i, val in enumerate(X_test)])

    def score(self, X_test, y_test):
        return sum(self.predict(X_test)==y_test) / len(y_test)


