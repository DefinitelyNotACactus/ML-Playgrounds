import numpy as np
from sklearn.svm import SVC

class SVF():
    def __init__(self, n_estimators=10, s_size=0.1, 
                 C=1.0, kernel='rbf', degree=3, 
                 gamma='scale', coef0=0.0, shrinking=True, 
                 probability=True, tol=0.001, cache_size=200, 
                 class_weight=None, verbose=False, max_iter=- 1, 
                 decision_function_shape='ovr', 
                 break_ties=False, random_state=None):

        self.n_estimators = n_estimators
        self.estimators = [None] * n_estimators
        self.s_size = s_size
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        
    def fit(self, X, y):
        if self.random_state is not None: np.random.seed(self.random_state)
        for i in range(self.n_estimators):
            estimator_random_state = self.random_state
            if estimator_random_state is not None: estimator_random_state += i
            # Sample the dataset
            sample_idx = np.random.choice(len(X), round(len(X) * self.s_size), replace=False)
            # Create the estimator
            self.estimators[i] = SVC(C=self.C, kernel=self.kernel, degree=self.degree, 
                        gamma=self.gamma, coef0=self.coef0, shrinking=self.shrinking, 
                        probability=self.probability, tol=self.tol, cache_size=self.cache_size, 
                        class_weight=self.class_weight, verbose=self.verbose, max_iter=self.max_iter, 
                        decision_function_shape=self.decision_function_shape, 
                        break_ties=self.break_ties, random_state=estimator_random_state)
            # Fit with the sampled dataset
            self.estimators[i].fit(X[sample_idx], y[sample_idx])
        
    def predict(self, X):
        y = np.zeros(len(X))
        for xn, i in zip(X, range(len(X))):
            # Assuming this is a binary classification problem (e.g., 1 for positive, 0 otherwise)
            positive_votes, negative_votes = 0, 0
            for estimator in self.estimators:
                pred = estimator.predict(xn.reshape(1, -1))
                if pred == 1: positive_votes += 1
                else: negative_votes += 1
            if positive_votes > negative_votes: y[i] = 1
            else: y[i] = 0 # In case of a tie, y[i] = 0, for now
                
        return y
    
    def decision_function(self, X):
        y = np.zeros(len(X))
        for xn, i in zip(X, range(len(X))):
            yn = 0
            for estimator in self.estimators: yn += estimator.decision_function(xn.reshape(1, -1))
            y[i] = yn / self.n_estimators
        
        return y