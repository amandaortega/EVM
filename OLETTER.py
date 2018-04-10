from config import qtas_letras
from config import qtas_letras_inicio
from config import sigma
from config import tau

from lib import base_original
from lib import calcula_delta
from lib import calcula_F1_score
from lib import plota_PCA
from lib import proximas_amostras
from lib import VR

from util import test_EVM
from util import train_EVM

import numpy as np

# obtendo e treinando o modelo com 15 letras aleatórias
(X_train, y_train, X_test, y_test, X_restante, y_restante, classes_conhecidas) = base_original("letter-recognition.csv", qtas_letras, qtas_letras_inicio)
(EVs_psi, EVs_X, EVs_y) = train_EVM(X_train, y_train, tau, sigma)

#plota_PCA(X_train, y_train, EVs_X, EVs_y)
VR(y_train, EVs_y)

# todas as classes de teste participaram do treinamento
Cr = qtas_letras_inicio
Ct = qtas_letras_inicio
Ce = qtas_letras_inicio
delta = calcula_delta(Ct, Cr, Ce)

# classifica as amostras de teste 
y_chapeu = test_EVM(EVs_psi, EVs_X, EVs_y, X_test, delta)

print(calcula_F1_score(y_test, y_chapeu))

classes_disponiveis = set(range(0, qtas_letras)).difference(classes_conhecidas)

# testa o modelo para as amostras das classes que vão sendo inseridas incrementalmente
for letra in range(0, qtas_letras - qtas_letras_inicio):
    # obtém as amostras das classes não treinadas a serem testadas
    (_, _, X_test_novo, _, X_restante, y_restante, letra_sorteada) = proximas_amostras(X_restante, y_restante, classes_disponiveis)

    # remove a letra sorteada do conjunto de letras disponíveis
    classes_disponiveis = classes_disponiveis.difference(letra_sorteada)

    # as amostras das classes que não foram usadas no treinamento deve sem classificadas como desconhecidas (-1)
    y_test_novo = - np.ones(X_test_novo.shape[0])

    # testa todas as amostras de teste das classes sorteadas, para produzir um único valor de F1-score
    X_test = np.concatenate((X_test, X_test_novo), axis=0)
    y_test = np.concatenate((y_test, y_test_novo), axis=0)  

    # atualiza as variáveis com mais uma classe inserida
    Cr += 1
    Ce +=1  
    delta = calcula_delta(Ct, Cr, Ce)

    # classifica as amostras de teste
    y_chapeu = test_EVM(EVs_psi, EVs_X, EVs_y, X_test, delta)

    print(calcula_F1_score(y_test, y_chapeu))