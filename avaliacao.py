import math
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

# fixa seed (p/ permitir replicacao)
random_state = 42

# carrega dados
df = pd.read_csv('dataset.csv')

# deixa apenas variaveis quantitativas (i.e., descarta colunas de texto)
df = df[['preco_total', 'area_util', 'local', 'vagas']]

# descarta outliers
df = df[df['area_util'] < 50000]
df = df[df['preco_total'] < 70000000]

# cria amostra ficticia (imovel que queremos preficicar)
x_new = pd.DataFrame({'preco_total': [1.0], 'area_util': [75], 'local': ['Noroeste'], 'vagas': [2]})
df = df.append(x_new)

# lineariza relacao entre preco e area util
df['preco_total'] = df['preco_total'].map(lambda x: math.log(x))
df['area_util'] = df['area_util'].map(lambda x: math.log(x))

# checa se relacoes fazem sentido
#df.plot.scatter(x = 'area_util', y = 'preco_total')
#df.plot.scatter(x = 'vagas', y = 'preco_total')
#plt.show()

# recodifica local
new_locals = {
    'Asa Sul': 'asa_sul',
    'Asa Norte': 'asa_norte',
    'Goiás': 'goias',
    'Águas Claras': 'satelite',
    'Taguatinga': 'satelite',
    'Guará': 'satelite',
    'Sudoeste': 'sudoeste',
    'Noroeste': 'noroeste',
    'Lago Norte': 'lago_norte',
    'Samambaia': 'satelite',
    'Ceilândia': 'satelite',
    'Centro': 'outros', # melhorar isso aqui depois (tem de tudo)
    'Setor De Industrias': 'asa_sul', # eh quase tudo SIG
    'Sobradinho': 'satelite',
    'Núcleo Bandeirante': 'satelite',
    'Riacho Fundo': 'satelite',
    'Vicente Pires': 'satelite',
    'Park Sul': 'satelite',
    'Recanto das Emas': 'satelite',
    'Lago Sul': 'lago_sul',
    'Gama': 'satelite',
    'Setor De Industria Graficas': 'asa_sul',
    'Setor Habitacional Jardim Botânico': 'satelite',
    'Octogonal': 'octogonal',
    'Planaltina': 'satelite',
    'Cruzeiro': 'cruzeiro',
    'Santa Maria': 'satelite',
    'São Sebastião': 'satelite',
    'Setor Da Industria E Abastecimento': 'outros',
    'Zona Industrial': 'outros',
    'Paranoá': 'satelite',
    'Setor De Autarquias Sul': 'asa_sul',
    'Setor Comercial Sul': 'asa_sul',
    'Setor Bancario Sul': 'asa_sul',
    'Setores Complementares': 'outros',
    'Park Way': 'outros',
    'Candangolândia': 'satelite',
    'Setor De Radio E Televisao Sul': 'asa_sul',
    'Taquari': 'satelite',
    'Setor Hoteleiro Sul': 'asa_sul',
    'Setor de Múltiplas Atividades Sul': 'outros',
    'Setor de Armazenagem e Abastecimento Norte': 'outros',
    'Setor Hospitalar Local Sul': 'asa_sul',
    'Zona Civico-administrativa': 'asa_sul',
    'Setor de Grandes Áreas Norte': 'asa_norte',
    'Setor De Clubes Esportivos Sul': 'lago_sul',
    'Zona Rural': 'outros',
    'Setor De Diversoes Norte': 'asa_norte',
    'Superquadra Sudoeste': 'sudoeste',
    'Setor de Mansões Dom Bosco': 'outros',
    'Setor Bancario Norte': 'asa_norte',
    'Setor Comercial Norte': 'asa_norte',
    'Setor De Oficinas Norte': 'asa_norte',
    'Setor Hoteleiro Norte': 'asa_norte',
    'Setor de Hotéis e Turismo Norte': 'asa_norte'
    }
df['local'] = df['local'].map(lambda x: new_locals[x])

# checa distribuicao dos imoveis por regiao
print(df['local'].value_counts())

# cria dummies p/ locais
locais = pd.get_dummies(df['local'], prefix = 'local', drop_first = True)
for col in locais.columns:
    df[col] = locais[col]
del df['local']

# extrai row c/ amostra ficticia
x_new = df[df['preco_total'] == 0]
del x_new['preco_total']
x_new = x_new.values
df = df[df['preco_total'] > 0]

# separa X e Y
y = df['preco_total'].values
del df['preco_total']
X = df.values

# aleatoriza ordem das amostras
X, y = shuffle(X, y, random_state = random_state)

# instancia modelo
#model = RandomForestRegressor(n_estimators = 1000, n_jobs = -1)
#model = SVR(kernel = 'poly', gamma = 'scale', epsilon = 0.001, C = 25)
#model = LinearRegression()
#model = AdaBoostRegressor(LinearRegression(), loss = 'exponential', n_estimators = 500)
model = GradientBoostingRegressor(loss = 'quantile', alpha = 0.5, n_estimators = 1000, random_state = random_state)
#model = XGBRegressor(n_estimators = 1000, booster = 'gbtree', reg_lambda = 0.5) # melhor desempenho

# treina modelo e gera estimativas
yhat = cross_val_predict(model, X, y, cv = 10)

# poe precos estimados na escala original (R$)
yhat_reais = np.exp(yhat)

# poe precos observados na escala original (R$)
y_reais = np.exp(y)

# calcula erros (R$)
erros = yhat_reais - y_reais

# calcula erro absoluto mediano (R$)
erro_absoluto_mediano = np.median(np.absolute(erros))
print('erro absoluto mediano:', erro_absoluto_mediano)

# calcula erros relativos (erro / valor observado do imovel)
erros_relativos = erros / y_reais
erro_relativo_mediano = np.median(np.absolute(erros_relativos))

# calcula erro relativo mediano
print('erro relativo mediano:', erro_relativo_mediano)

# plota algumas relacoes
df = pd.DataFrame([])
df['area_util'] = X[:,0]
df['area_util'] = df['area_util'].map(lambda x: math.exp(x))
df['yhat'] = yhat
df['yhat_reais'] = yhat_reais
df['y_reais'] = y_reais
df['erro'] = erros
df['erro_relativo'] = erros_relativos
#df.plot.scatter(x = 'y_reais', y = 'yhat_reais')
#df.plot.scatter(x = 'yhat_reais', y = 'erro')
#df.plot.scatter(x = 'area_util', y = 'erro')
#plt.show()

# cria modelos p/ gerar estimativas e intervalos
model_lower = GradientBoostingRegressor(loss = 'quantile', alpha = 0.25, n_estimators = 1000, random_state = random_state)
model_mid = GradientBoostingRegressor(loss = 'quantile', alpha = 0.5, n_estimators = 1000, random_state = random_state)
model_upper = GradientBoostingRegressor(loss = 'quantile', alpha = 0.75, n_estimators = 1000, random_state = random_state)
model_lower.fit(X, y)
model_mid.fit(X, y)
model_upper.fit(X, y)

# estima valores p/ nova amostra
yhat_lower = model_lower.predict(x_new)
yhat_mid = model_mid.predict(x_new)
yhat_upper = model_upper.predict(x_new)
print(np.exp(yhat_lower), np.exp(yhat_mid), np.exp(yhat_upper))
