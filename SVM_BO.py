import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import accuracy_score
from scipy.stats import norm

class SVM_BO():
    def __init__(self, n_samples=3, min_pct_initial=0.1, max_pct_initial=0.1, 
                 n_candidates=100, min_pct=0.1, max_pct=0.5, max_itr=100, epsilon=0.002,
                 C=1.0, kernel='rbf', degree=3, 
                 gamma='scale', coef0=0.0, shrinking=True, 
                 probability=False, tol=0.001, cache_size=200, 
                 class_weight=None, verbose=False, max_iter=- 1, 
                 decision_function_shape='ovr', 
                 break_ties=False, random_state=None):
        # Model
        self.model = None
        # BO Parameters
        self.n_samples = n_samples
        self.min_pct_initial = min_pct_initial
        self.max_pct_initial = max_pct_initial
        self.n_candidates = n_candidates
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.max_itr = max_itr
        self.epsilon = epsilon
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
    
    def get_initial_params(self, X, y, n_samples=10, min_pct=0.1, max_pct=0.5):
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = y.values
        X_params = []
        y_params = [0] * n_samples
        for i in range(n_samples):
            pct = np.random.uniform(min_pct, max_pct)
            X_params.append(np.zeros(len(X)))
            sample = np.random.choice(len(X), size=int(np.ceil(len(X) * pct)), replace=False)
            X_params[i][sample] = 1
            y_params[i], model = self.obj_function(X[sample], y[sample], X, y)

        return X_params, y_params

    # Função objetivo: Diferença absoluta entre a acurácia do modelo em S e U
    def obj_function(self, S_X, S_y, U_X, U_y):
        model = SVC(C=self.C, kernel=self.kernel, degree=self.degree, 
                    gamma=self.gamma, coef0=self.coef0, shrinking=self.shrinking, 
                    probability=self.probability, tol=self.tol, cache_size=self.cache_size, 
                    class_weight=self.class_weight, verbose=self.verbose, max_iter=self.max_iter, 
                    decision_function_shape=self.decision_function_shape, 
                    break_ties=self.break_ties, random_state=self.random_state)
        model.fit(S_X, S_y)
        pred_U = model.predict(U_X)
        pred_S = model.predict(S_X)
        
        return abs(accuracy_score(S_y, pred_S) - accuracy_score(U_y, pred_U)), model

    # Função substituta
    def surrogate_function(self, model, X): return model.predict(X, return_std=True)

    #Função de aquisição: probabilidade de melhora
    def acquisition_function(self, model, X, candidates):
        yhat, std = self.surrogate_function(model, X)
        best = max(yhat)
        
        mean, std = self.surrogate_function(model, candidates)
        
        improvement_probability = norm.cdf((mean - best) / (std + 1E-9))
        
        return improvement_probability

    # Otimizar a função de aquisição, selecionando o candidato que maximiza a probabilidade de melhora
    def opt_acquisition(self, X, y, model, len_dataset, n_candidates=100, min_pct=0.1, max_pct=0.5):
        # Criar os pontos candidatos
        candidates = np.zeros((n_candidates, len_dataset))
        candidates_idx = []
        for i in range(n_candidates):
            pct = np.random.uniform(min_pct, max_pct)
            sample = np.random.choice(len_dataset, size=int(np.ceil(len_dataset * pct)), replace=False)
            candidates[i][sample] = 1
            candidates_idx.append(sample)
            
        scores = self.acquisition_function(model, X, candidates)
        best = np.argmax(scores)
        
        return candidates[best], candidates_idx[best]

    def fit(self, X, y):
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = y.values
        y = np.array([1 if yn == 1 else -1 for yn in y])
        X_params, y_params = self.get_initial_params(X, y, self.n_samples, self.min_pct_initial, self.max_pct_initial)
        # Criar o modelo GP sobre os pontos inicialmente conhecidos
        gp = GaussianProcessRegressor()
        gp.fit(X_params, y_params)
        best_score, best_params, best_params_idx, best_SX, best_Sy, best_model = np.inf, None, None, None, None, None
        for i in range(self.max_itr):
            # Selecionar um ponto
            q, q_idx = self.opt_acquisition(X_params, y_params, gp, len(X), self.n_candidates, self.min_pct, self.max_pct)
            # Computar o valor da função objetivo nesse ponto
            q_y, model_q = self.obj_function(X[q_idx], y[q_idx], X, y)
            # Checar se encontramos um novo mínimo para a função objetivo
            if q_y < best_score: 
                (best_score, best_params, best_params_idx, 
                    best_SX, best_Sy, best_model) = q_y, q, q_idx, X[q_idx], y[q_idx], model_q
                if best_score < self.epsilon: break
            # Adicionar os dados obtidos ao que já conhecemos
            X_params = np.append(X_params, [q], axis=0)
            y_params.append(q_y)
            # Retreinar o processo gaussiano
            gp.fit(X_params, y_params)
        # Retorne o melhor conjunto de parâmetros
        self.model = best_model
        S_X, S_y = [], []
        U_X, U_y = [], []
        for i, (xn, yn) in enumerate(zip(X, y)):
            if i in best_params_idx: 
                S_X.append(xn)
                S_y.append(yn)
            else: 
                U_X.append(xn)
                U_y.append(yn)

        return best_model, i, np.array(S_X), np.array(U_X), np.array(S_y), np.array(U_y)
        
    def predict(self, X):
        return self.model.predict(X)