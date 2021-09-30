import numpy as np
import pandas as pd
import forestci as fci
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

# set seed
random_state = 451

# load data
df = pd.read_csv('wimoveis.csv')
 
# keep only certain features
df = df[[
    'anuncio_id',
    'preco_total', 
    'area_util', 
    'local', 
    'iptu', 
    'condominio', 
    'quartos', 
    'suites', 
    'banheiros', 
    'vagas', 
    'churrasqueira',
    'idade',
    'brinquedoteca',
    'tv',
    'piscina',
    'playground',
    'sauna',
    'academia',
    'portaria',
    'jogos',
    'festas',
    'andares',
    ]]
del df['anuncio_id']

# impute median values for missing condo fees and property tax
for var in ['iptu', 'condominio']:
    median = df[df[var] > 0][var].median()
    df[var] = df[var].fillna(median)

# drop outliers
df = df[df['area_util'] < 50000]
df = df[df['preco_total'] < 10000000]
df = df[df['condominio'] < 1500000]
df = df[df['iptu'] < 20000]
df = df[df['vagas'] < 10]

# take the log of all quantitative non-image features
for var in ['preco_total', 'area_util', 'iptu', 'condominio']:
    df[var] = df[var].map(lambda x: np.log(x))

# dummify location
locais = pd.get_dummies(df['local'], prefix = 'local', drop_first = True)
for col in locais.columns:
    df[col] = locais[col]
del df['local']

# split X and Y
y = df['preco_total'].values
del df['preco_total']
X = df.values

# shuffle sample order
X, y = shuffle(X, y, random_state = random_state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = random_state)

# instantiate model
model = RandomForestRegressor(n_estimators = 1000, criterion = 'mse', n_jobs = -1, random_state = random_state)

# train model
yhat = cross_val_predict(model, X_train, y_train, cv = 10)

# put estimated prices back in R$
yhat_reais = np.exp(yhat)

# put observed prices back in R$
y_reais = np.exp(y_train)

# compute errors
errors = yhat_reais - y_reais

# compute median absolute error
median_abs_error = np.median(np.absolute(errors))
print('median absolute error (in R$):', median_abs_error)

# compute proportional error (error / asking price)
proportional_errors = errors / y_reais
median_prop_error = np.median(np.absolute(proportional_errors))
mean_prop_error = np.mean(np.absolute(proportional_errors))
print('median absolute error (in %):', median_prop_error)
print('mean absolute error (in %):', mean_prop_error)
