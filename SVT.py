from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

class SVT(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma='scale', C=1.0, kernel='rbf', degree=3, random_state=42):
        # Parâmetros do SVM
        # TODO: Adicionar os outros parâmetros
        self.C = C
        self.kernel = kernel
        self.random_state = random_state
        self.gamma = gamma
        self.degree = degree

    def truncate(self, number, digits) -> float:
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper
    
    def fit(self, X, y):
        # Treina o SVM
        model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree)
        model.fit(X, y)
        predict = model.predict(X)
        # Pegar os vetores de suporte
        self.vs = [support for support, coef in zip(model.support_, model.dual_coef_[0])]# if abs(coef) != self.C]
        self.n_vs = len(self.vs)
        #alphas = svm.dual_coef_[0]
        #param = SVM_Parameters(model, X, y)
        #for support in svm.support_:
            #vs.append(support)
            #if predict[support] == y[support]:
            #    vs.append(support)
            #lh = param.compute_left_hand(X[support], y[support])
            #try:
            #    if self.truncate(lh, 3) < 1: # Vetor de suporte fora da margem
            #        if predict[support] == y[support]:
            #            vs.append(support)
            #            #print('VS fora da margem bem classificado')
            #        else: print('VS fora da margem mal classificado')
            #    else: 
                    #print('VS na margem')
            #        vs.append(support)
            #except:
            #    vs.append(support)
            
        cart = DecisionTreeClassifier(random_state=0)
        #path = cart.cost_complexity_pruning_path(X[vs], y[vs])

        #param_grid = {'ccp_alpha': path.ccp_alphas}

        #K = 5
        #if len(vs) < 5: K = len(vs)

        #self.model = GridSearchCV(estimator=cart, param_grid=param_grid, cv=K, verbose=0, n_jobs=-1)
        self.model = cart
        self.model.fit(X[self.vs], y[self.vs])
            
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)