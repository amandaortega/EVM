import sklearn.metrics
import numpy as np
import libmr

def fit_psi(X, y, tau, Cl):
    # obtém as amostras da classe atual
    l = np.argwhere(y == Cl).reshape(-1)
    X_l = X[l]

    # obtém as amostras das outras classes
    m = np.argwhere(y != Cl).reshape(-1)
    X_m = X[m]

    # calcula os pares de distâncias entre as amostras da classe
    # atual, X_l, e as amostras das outras classes, X_m
    D = sklearn.metrics.pairwise.pairwise_distances(X_l, X_m)

    psi = []

    # para cada amostra pertencente à classe Cl, estima os
    # parâmetros shape e scale com base na metade da distância
    # das tau amostras mais próximas não pertencentes a Cl
    for i in range(X_l.shape[0]):
        mr = libmr.MR()
        mr.fit_low(1/2 * D[i], tau)
        psi.append(mr)
    
    return np.array(psi)

def probabilidade(psi, distancia):
    return psi.w_score(distancia)

def set_cover(X_l, psi, sigma):
    D = sklearn.metrics.pairwise.pairwise_distances(X_l)

    Nl = X_l.shape[0]
    U = range(Nl)
    S = np.zeros((Nl, Nl))

    # percorre todas as amostras e verifica se probabilidade
    # gerada pela função psi entre a distância entre cada par 
    # de pontos é maior ou igual a sigma
    for i in U:
        for j in U:
            if probabilidade(psi[i], D[i, j]) >= sigma:
                S[i, j] = 1
    
    C = []
    I = []

    # enquanto os pontos representados pelos valores 
    # extremos não abrangerem o universo das amostras
    while (set(C) != set(U)):
        # manipulações necessárias para que a matriz
        # diferenca contenha 1 apenas quando há um 
        # ponto ainda não coberto e que é representado 
        # por um valor extremo candidato
        aux = np.zeros(Nl)
        aux[C] = 1
        diferenca = S - np.matlib.repmat(aux, Nl, 1)
        diferenca[I] = np.zeros((len(I), Nl))
        diferenca[:, I] = np.zeros((Nl, len(I)))
        diferenca = np.clip(diferenca, 0, 1)
        
        #obtém o valor extremo que representa a maior quantidade 
        # de pontos ainda não cobertos
        ind = np.argmax(np.sum(diferenca, axis=1), axis=0)

        # acrescenta os novos pontos cobertos
        C = np.append(C, np.asarray(np.where(S[ind])).reshape(-1))
        C = C.astype(int)
        C = np.unique(C)

        # acrescenta o novo valor extremo ao conjunto
        I.append(ind)
    
    return np.asarray(I).reshape(-1).tolist()

def test_EVM(EVs_psi, EVs_X, EVs_y, X_test, delta):
    # número de amostras de teste e de vetores extremos
    M = X_test.shape[0]
    L = EVs_X.shape[0]

    # y_chapeu contém as predições que serão geradas pelo modelo; se
    # nenhum vetor extremo gerar uma probabilidade de pertencimento à classe
    # maior ou igual a delta, a amostra será classificada como não pertencente
    # a nenhuma das classes vistas durante o treinamento, representada por -1
    y_chapeu = - np.ones(M)

    # matriz de distância entre os vetores extremos e as amostras de teste
    D = sklearn.metrics.pairwise.pairwise_distances(EVs_X, X_test)

    # para cada amostra de teste...
    for m in range(0, M):
        maior_probabilidade = 0

        # ...percorre os vetores extremos, verificando se é possível englobar
        # a amostra à distribuição representada por cada EV 
        for l in range(0, L):
            prob = probabilidade(EVs_psi[l], D[l, m])

            # se a probabilidade de inclusão foi maior que delta e maior que 
            # a maior probabilidade obtida até então, armazena a probabilidade
            # e classifica a amostra de teste com a mesma classe referente ao EV
            if prob >= delta and prob > maior_probabilidade:
                maior_probabilidade = prob
                y_chapeu[m] = EVs_y[l]

    return y_chapeu

def train_EVM(X, y, tau, sigma):
    # conjunto das classes existentes
    C = np.unique(y)

    # listas com as informações dos vetores extremos para cada classe
    EVs_psi = []
    EVs_X = []
    EVs_y = []

    # para cada classe no conjunto
    for Cl in C:
        # treina a amostra, encaixando um vetor de funções de distribuição
        # de probabilidade para cada classe
        psi = fit_psi(X, y, tau, Cl)

        # obtém as amostras da classe atual
        l = np.argwhere(y == Cl).reshape(-1)
        X_l = X[l]
        y_l = y[l]

        # elima os vetores de funções de distribuição de probabilidade
        # desnecessários, deixando apenas os vetores extremos
        I = set_cover(X_l, psi, sigma)
        
        # acrescenta as informações dos vetores extremos às listas
        if len(EVs_X) > 0:
            EVs_X = np.concatenate((EVs_X, X_l[I]), axis=0)
            EVs_y = np.concatenate((EVs_y, y_l[I]), axis=0)
            EVs_psi = np.concatenate((EVs_psi, psi[I]), axis=0)            
        else:
            EVs_X = X_l[I]
            EVs_y = y_l[I]            
            EVs_psi = psi[I]
    
    return (EVs_psi, np.asarray(EVs_X), np.asarray(EVs_y))