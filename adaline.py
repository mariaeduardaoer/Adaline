import numpy as np
import pandas as pd

class Adaline:
    
    def __init__(self, input_values, output_values, learning_rate, precision, activation_function):
        ones_column = np.ones((len(input_values), 1)) * (-1)
        self.input_values = np.append(ones_column, input_values, axis=1)
        self.output_values = output_values
        self.learning_rate = learning_rate
        self.precision = precision
        self.activation_function = activation_function
        self.W = np.random.rand(self.input_values.shape[1])
        print(f'Inicial W: {self.W}')
       
        
    def train(self):
        epochs = 1
        eqm_values = []
       

        while True:
            eqm_anterior = self.erro_quad_medio(self.W)
          
            for x, d in zip(self.input_values, self.output_values):
                 u = np.dot(x, self.W)
                 self.W = self.W + self.learning_rate * (d - u) * x    
                 
                 # y = self.activation_function.g(u)
                 # print(f'Input: {x}, Output: {y}, Expected: {d}')
                
                    
            epochs += 1
            eqm_atual = self.erro_quad_medio(self.W)
            eqm_values.append(eqm_atual)

            if abs(eqm_atual - eqm_anterior) <= self.precision:
                break
            
        
        
        print(f'EPOCHS: {epochs}')    
        print(f'Final W: {self.W}')
        print('')
        return eqm_values
    

    def erro_quad_medio(self, W):
        p = len(pd.read_csv('database/dataset-treinamento_ADALINE.csv'))
        eqm = 0
        
        for x, d in zip(self.input_values, self.output_values):
            u = np.dot(x, self.W)
            eqm = eqm + (d - u) ** 2
            
        eqm = eqm/p
        return eqm
        
    
    def evaluate(self, input_values):
        input_values= np.append([[-1]],[input_values], axis=1) # -1 na primeira posição
        u = np.dot(input_values, self.W)
        return self.activation_function.g(u)