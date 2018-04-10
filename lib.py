from collections import Counter
from collections import OrderedDict

from config import codigo_A
from config import qtas_letras

from random import sample
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

import csv
import matplotlib.pyplot as plt
import numpy as np

def base_original(nome_arquivo, tamanho_total, tamanho_inicial):
    # lê a base de dados
    reader = csv.reader(open(nome_arquivo, "r"), delimiter=",")
    x = list(reader)
    base = np.array(x)

    # separa o conjunto de features X da saída y
    X = base[:, 1:]
    y = base[:, 0]

    # mapeia as letras em números e converte para um vetor de inteiros
    for i in range(0, y.size):
        y[i] = ord(y[i]) - codigo_A
    y = y.astype(int)

    # sorteia as classes que serão usadas durante o treinamento
    letras_treinamento = sample(range(0, tamanho_total), k=tamanho_inicial)

    return separa_base(X, y, letras_treinamento)

def calcula_delta(Ct, Cr, Ce):
    return 1/2 * (1 - np.sqrt(2 * Ct / (Cr + Ce)))

def calcula_F1_score(y, y_chapeu):
    return f1_score(y, y_chapeu, average='micro')  

def plota_PCA(X, y, EVs_X, EVs_y):
    X_reduzido = PCA(n_components=2).fit_transform(X)
    EVs_X_reduzido = PCA(n_components=2).fit_transform(EVs_X)

    x_min, x_max = X_reduzido[:, 0].min() - .5, X_reduzido[:, 0].max() + .5
    y_min, y_max = X_reduzido[:, 1].min() - .5, X_reduzido[:, 1].max() + .5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(X_reduzido[:, 0], X_reduzido[:, 1], c=y, edgecolor='k')
    plt.scatter(EVs_X_reduzido[:, 0], EVs_X_reduzido[:, 1], c=EVs_y, edgecolor='k', 
        marker="s")    
    plt.xlabel('1a componente')
    plt.ylabel('2a componente')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()

def proximas_amostras(X, y, classes_disponiveis):
    classe_sorteada = sample(classes_disponiveis, 1)

    return separa_base(X, y, classe_sorteada)

def separa_base(X, y, letras_treinamento):
    # sorteia 80% dentre as amostras cujas letras foram sorteadas para a 
    # etapa de treinamento
    amostras_classes_sorteadas = np.argwhere(np.isin(y, letras_treinamento).reshape(-1)).reshape(-1)
    amostras_treinamento = sample(amostras_classes_sorteadas.tolist(), k=(round(amostras_classes_sorteadas.size * 0.8)))

    # obtém a base de dados de treinamento
    X_train = X[amostras_treinamento]
    y_train = y[amostras_treinamento]

    # obtém as amostras de teste das classes sorteadas
    amostras_teste = np.array(list(set(amostras_classes_sorteadas).difference(set(amostras_treinamento))))

    # obtém a base de dados de teste
    X_test = X[amostras_teste]
    y_test = y[amostras_teste]

    # retira da base original as amostras das classes sorteadas
    np.delete(X, amostras_classes_sorteadas, 0)
    np.delete(y, amostras_classes_sorteadas, 0) 

    return (X_train, y_train, X_test, y_test, X, y, set(y_train))        

def VR(y, EVs_y):
    # conta a frequência de ocorrência de cada classe da amostra original e dos 
    # valores extremos
    frequencia_original = Counter(y)
    frequencia_EV = Counter(EVs_y)

    VRs = {}

    # para cada letra, calcula o VR (vector ratio), quando possível
    for letra in range(0, qtas_letras):
        chave = chr(letra + codigo_A)
        # a letra não foi considerada na base atual
        if frequencia_original[letra] == 0:
            VRs[chave] = '-'
        else:
            VRs[chave] = round(frequencia_EV[letra] / frequencia_original[letra], 2)
    
    # ordena por letra
    VRs = OrderedDict(sorted(VRs.items()))

    print('VRs por letra:')
    print(VRs)
    print('VR geral: ', round(len(EVs_y) / len(y), 2))