import numpy as np

from math import trunc

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from SVM_Parameters import SVM_Parameters

# Função pra truncar um float, de forma que ignore o pequeno ruído de quando se está na margem
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return trunc(stepper * number) / stepper

def train_SVM_batch(X, y, S_size, steps, kernel_input, c_input, gamma_input, degree_input):
    y_new = [1 if yi == 1 else -1 for yi in y]
    S_X, U_X, S_y, U_y = train_test_split(X, y_new, test_size=1-S_size, random_state=42)
    i = 0
    in_S = {}
    S_to_U = np.array([-1 for idx in range(len(S_X))])
    while i < steps:
        model = SVC(kernel=kernel_input, C=float(c_input), gamma=gamma_input, degree=float(degree_input))
        model.fit(S_X, S_y)

        param = SVM_Parameters(model, S_X, S_y, margin=False)
        # Retira de S todos os pontos que não são VS do alpha da iteração anterior
        r = range(len(S_X))
        if i > 1:
            to_remove_idx = np.array([j for j in r if j not in model.support_]) # Indices que não são VS
            to_remove_idx_in_S = []
            for to_remove in to_remove_idx:
                if S_to_U[to_remove] != -1: # Checar se o elemento é um VS originário de S
                    # Atualiza o dicionario in_S para informar que o elemento não faz mais parte de S
                    # S_to_U mapeia um indice em S para o respectivo indice em U
                    in_S[S_to_U[to_remove]] = 'No'

            if len(to_remove_idx) > 0: S_to_U = np.delete(S_to_U, to_remove_idx, axis=0)
        S_X = np.array([x for x, j in zip(S_X, r) if j in model.support_])
        S_y = np.array([y for y, j in zip(S_y, r) if j in model.support_])
        
        if len(U_X) == 0: break
        predict = model.predict(U_X)
        vs_wrong_pred = vs_violate = 0
        VS_indexes = []
        # Lista para consulta se um elemento de U está em S, inicialmente todos de U não estão em S
        if i == 2: in_S = {element: 'No' for element in range(len(U_X))}
        i_x = 0
        
        S_X = S_X.tolist()
        S_y = S_y.tolist()
        S_to_U = S_to_U.tolist()
        for xn, yn, prediction in zip(U_X, U_y, predict):
            if i > 1 and in_S[i_x] == 'Yes': continue
            if prediction != yn:
                S_X.append(xn)
                S_y.append(yn)
                in_S[i_x] = 'Yes'
                S_to_U.append(i_x)
                VS_indexes.append(i_x)
                vs_wrong_pred += 1
            else:
                # Considerar epsilon = 10**-3
                left_hand = param.compute_left_hand(xn, yn)
                if left_hand == np.inf or truncate(left_hand, 3) <= 1:
                    S_X.append(xn)
                    S_y.append(yn)
                    in_S[i_x] = 'Yes'
                    S_to_U.append(i_x)
                    VS_indexes.append(i_x)
                    vs_violate += 1
            i_x += 1
            
        S_X = np.array(S_X)
        S_y = np.array(S_y)
        S_to_U = np.array(S_to_U)

        if len(VS_indexes) == 0: break
        elif i == 1:
            U_X = np.delete(U_X, VS_indexes, axis=0)
            U_y = np.delete(U_y, VS_indexes, axis=0)

        i += 1

    return model, i, S_X, U_X, S_y, U_y
