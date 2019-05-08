import math
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_predict

# carrega dados
df = pd.read_csv('wimoveis.csv')

# deixa apenas variaveis quantitativas (i.e., descarta colunas de texto)
df = df[['preco_total', 'area_util', 'local', 'vagas']]

# descarta outliers
df = df[df['area_util'] < 50000]
df = df[df['preco_total'] < 70000000]

# lineariza relacao entre preco e area util
df['preco_total'] = df['preco_total'].map(lambda x: math.log(x))
df['area_util'] = df['area_util'].map(lambda x: math.log(x))

# checa se relacoes fazem sentido
#df.plot.scatter(x = 'area_util', y = 'preco_total')
#df.plot.scatter(x = 'vagas', y = 'preco_total')

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
print(df['local'].value_counts())

# cria dummies p/ locais
locais = pd.get_dummies(df['local'], prefix = 'local', drop_first = True)
for col in locais.columns:
    df[col] = locais[col]
del df['local']

# separa X e Y
y = df['preco_total'].values
del df['preco_total']
X = df.values

# aleatoriza ordem das amostras
X, y = shuffle(X, y, random_state = 42)

# instancia modelo
#clf = RandomForestRegressor(n_estimators = 1000, n_jobs = -1)
#clf = SVR(kernel = 'poly', gamma = 'scale', epsilon = 0.001, C = 25)
#clf = LinearRegression()
#clf = AdaBoostRegressor(LinearRegression(), loss = 'exponential', n_estimators = 500)
#clf = GradientBoostingRegressor()
clf = XGBRegressor(n_estimators = 1000, booster = 'gbtree', reg_lambda = 0.5) # melhor desempenho

# treina modelo e gera estimativas
yhat = cross_val_predict(clf, X, y, cv = 10)

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
df['yhat'] = yhat
df['yhat_reais'] = yhat_reais
df['y_reais'] = y_reais
df['erro'] = erros
df['erro_relativo'] = erros_relativos
#df.plot.scatter(x = 'y_reais', y = 'yhat_reais')
#df.plot.scatter(x = 'yhat_reais', y = 'erro')
#df.plot.scatter(x = 'area_util', y = 'erro')
#plt.show()

# poe area util na escala original (m2)
df['area_util'] = df['area_util'].map(lambda x: math.exp(x))

# checa erro absoluto mediano por faixa de metragem
step = 1000
i = 0
while i < (10 * step):
    min_area, max_area = i, i + step
    subset = df[(df['area_util'] >= min_area) & (df['area_util'] < max_area)]
    erro_absoluto_mediano = subset['erro'].map(lambda x: abs(x)).median()
    qtde = subset.shape[0]
    preco_mediano = subset['y_reais'].median()
    i += step
    print(' ')
    print('faixa:', min_area, 'm2 a', max_area, 'm2')
    print('erro absoluto mediano:', round(erro_absoluto_mediano, 2))
    print('qtde:', qtde)
    print('preco mediano:', round(preco_mediano, 2))
