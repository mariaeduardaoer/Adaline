import pandas as pd
import matplotlib.pyplot as plt

from activation_functions import SignFunction
from adaline import Adaline


# Database dataset-treinamento:
dataset = pd.read_csv('database/dataset-treinamento_ADALINE.csv')
X = dataset.iloc[:,0:4].values 
d = dataset.iloc[:,4:].values


adaline = Adaline(X, d, 0.0025, 10**(-6), SignFunction)  # entrada, saída, taxa de ativação, precisão e função de ativação
values_eqm = adaline.train()



# Database dataset-teste:
dataset = pd.read_csv('database/dataset-teste_ADALINE.csv')
X_teste = dataset.iloc[:,0:4].values


for x in X_teste:
    y = adaline.evaluate(x)
    print(f'Input: {x},Output: {y}')    


# Plotando figura do problema
plt.plot(values_eqm)
plt.show()