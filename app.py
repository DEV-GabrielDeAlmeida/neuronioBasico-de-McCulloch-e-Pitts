import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Leitura do arquivo de dados
dados = pd.read_csv("C:/Users/usuário/Desktop/neurônio básico/dados.csv")

# Separa as features (entradas) e o rótulo (saída)
X = dados.iloc[:, :-1].values
y = dados.iloc[:, -1].values

# Converter os rótulos para o formato bipolar, por exemplo:
y = np.where(y == 0, -1, 1)

# Divide os dados em 2/3 para treinamento e 1/3 para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=94)

# 2. Inicializa os pesos aleatoriamente
n_features = X_train.shape[1]
np.random.seed(94) #<<<<<<<<<<<<<<<<<<<<<<
pesos = np.random.rand(n_features)

# Definindo o learning rate e número máximo de épocas
taxa_aprendizado = 0.1
max_epocas = 1000

# Função de ativação Bipolar
def ativacao_bipolar(x):
    return np.where(x >= 0, 1, -1)

# Treinamento do Perceptron
for epoca in range(max_epocas):
    erro_total = 0
    for xi, alvo in zip(X_train, y_train):
        # 3. Calcula a soma ponderada (agregação) (usando bias zero, Ɵ=0)
        soma_ponderada = np.dot(xi, pesos)
        
        # 4. Calcula a função de ativação
        saida = ativacao_bipolar(soma_ponderada)
        
        # Calcula o erro
        erro = alvo - saida
        erro_total += abs(erro)
        
        # 5. Atualiza os pesos
        pesos += taxa_aprendizado * erro * xi

    # 6. Critério de parada: se o erro total for zero, o treinamento convergiu
    if erro_total == 0:
        print(f"Treinamento convergiu na época {epoca+1}")
        break

# faz a predição com o Perceptron treinado
def prever(X):
    soma = np.dot(X, pesos)
    return ativacao_bipolar(soma)

# faz a predição no conjunto de teste
y_pred = prever(X_test)

# calcula a acurácia
acuracia = np.mean(y_pred == y_test) * 100
print(f"Acurácia no conjunto de teste: {acuracia:.2f}%")
