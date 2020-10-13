# Adaline

#### Projeto da cadeira de Sistemas Inteligentes

Utilizar o algoritmo de aprendizado regra Delta visando a classificação de padrões pelo
Adaline para o seguinte problema:



### Sistema de Gerenciamento Automático de duas Válvulas

Um sistema de gerenciamento automático de duas válvulas, situado a 500 metros de um processo
industrial, envia um sinal codificado constituído de quatro grandezas x1, x2, x3 e x4, as quais
são necessárias para seus acionamentos. Uma mesma via de comunicação é utilizada para acionar 
ambas as válvulas, sendo que o comutador localizado próximo a estas deve decidir se o sinal é
para a válvula A ou válvula B.

Entretanto, durante a comunicação, os sinais sofrem interferências que alteram o conteúdo das
informações originalmente transmitidas. Para contornar este problema, a equipe de engenheiros
e cientistas pretende treinar um Adaline para classificar os sinais ruidosos, cujo o objetivo
é então garantir ao sistema comutador se os dados devem ser encaminhados para o comando de 
ajuste da válvula A ou B.

Assim fundamentado nas medições de alguns sinais já com ruídos compilou-se o conjunto de
treinamento tomando-se por convenção o valor -1 para os sinais a serem encaminhados para o
ajuste da válvula A e o valor +1 para ajustes da válvula B.




#### Regra Delta:

W = W + learning_rate * (d - u) * x

