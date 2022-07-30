import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors

class CBD():
    def __init__(self, s_size=0.1, K=50, 
                 C=1.0, kernel='rbf', degree=3, 
                 gamma='scale', coef0=0.0, shrinking=True, 
                 probability=False, tol=0.001, cache_size=200, 
                 class_weight=None, verbose=False, max_iter=- 1, 
                 decision_function_shape='ovr', 
                 break_ties=False, random_state=None):
        # Model
        self.model = None
        # CBD Parameters
        self.K = K
        self.s_size = s_size
        # SVC Parameters
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
        
    def determine_scores(self, K, X, y):
        if K >= len(X): K = len(X) - 1
        nn = NearestNeighbors(n_neighbors=K+1).fit(X)
        distance, indices = nn.kneighbors(X)

        n = len(X)
        # Determine gamma
        gamma = 0
        counter = 0
        for i in range(n):
            nearest_opposite_neighbor_found = False
            for j in range(1, K+1):
                if y[i] != y[indices[i][j]]:
                    if nearest_opposite_neighbor_found is False:
                        nearest_opposite_neighbor_found = True
                        t_i = distance[i][j]
                    gamma += distance[i][j] - t_i
                    counter += 1
        gamma /= counter

        # Determine instance score
        scores = np.zeros(n)
        contributions = np.zeros(n)
        for i in range(n):
            nearest_opposite_neighbor_found = False
            for j in range(1, K+1):
                 if y[i] != y[indices[i][j]]:
                    if nearest_opposite_neighbor_found is False:
                        nearest_opposite_neighbor_found = True
                        t_i = distance[i][j]
                    scores[indices[i][j]] += np.exp(-1 * ((distance[i][j] - t_i) / gamma))
                    contributions[indices[i][j]] += 1

        for i in range(n): 
            if contributions[i] > 0: scores[i] /= contributions[i]

        return scores

    def fit(self, X, y):
        if self.random_state is None: self.random_state = np.random.seed()
        y = np.array([1 if yn == 1 else -1 for yn in y])
        scores = self.determine_scores(self.K, X, y)
        sorted_scores = sorted(range(len(X)), key = lambda x: scores[x], reverse=True)
        selected_indices = sorted_scores[:int(np.round(len(X) * self.s_size))]
        S_X = X[selected_indices]
        S_y = y[selected_indices]
        U_X = np.delete(X, selected_indices, axis=0)
        U_y = np.delete(y, selected_indices, axis=0)

        self.model = SVC(C=self.C, kernel=self.kernel, degree=self.degree, 
                    gamma=self.gamma, coef0=self.coef0, shrinking=self.shrinking, 
                    probability=self.probability, tol=self.tol, cache_size=self.cache_size, 
                    class_weight=self.class_weight, verbose=self.verbose, max_iter=self.max_iter, 
                    decision_function_shape=self.decision_function_shape, 
                    break_ties=self.break_ties, random_state=self.random_state)
        self.model.fit(S_X, S_y)

        return self.model, S_X, U_X, S_y, U_y
        
    def predict(self, X):
        return self.model.predict(X)