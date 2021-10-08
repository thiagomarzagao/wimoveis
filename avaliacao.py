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

# target apartment
x_new = pd.DataFrame({
    'area_util': [154.07], 
    'iptu': [2464.8], # https://www.wimoveis.com.br/propriedades/sqn-106-asa-norte-2955725723.html
    'condominio': [1230], # https://www.wimoveis.com.br/propriedades/sqn-106-asa-norte-2955725723.html
    'quartos': [3],
    'suites': [1],
    'banheiros': [1],
    'vagas': [1],
    'churrasqueira': [0],
    'idade': [55],
    'brinquedoteca': [0],
    'tv': [1],
    'piscina': [0],
    'playground': [0],
    'sauna': [0],
    'academia': [0],
    'portaria': [1],
    'jogos': [0],
    'festas': [0],
    'andares': [6],
    'local': ['Asa Norte'], 
    'preco_total': [1.42], 
    })
df = df.append(x_new)

# take the log of all quantitative non-image features
for var in ['preco_total', 'area_util', 'iptu', 'condominio']:
    df[var] = df[var].map(lambda x: np.log(x))

# dummify location
locais = pd.get_dummies(df['local'], prefix = 'local', drop_first = True)
for col in locais.columns:
    df[col] = locais[col]
del df['local']

# extract row w/ target apartment
x_new = df[df['preco_total'] == np.log(1.42)]
del x_new['preco_total']
x_new = x_new.values
df = df[df['preco_total'] > np.log(1.42)]

# split X and Y
y = df['preco_total'].values
del df['preco_total']
X = df.values

# shuffle sample order
X, y = shuffle(X, y, random_state = random_state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = random_state)

# instantiate model
n_trees = 1000
model = RandomForestRegressor(n_estimators = n_trees, criterion = 'mse', n_jobs = -1, random_state = random_state)

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

'''
df_output = pd.DataFrame([])
df_output['yhat'] = yhat
df_output['yhat_reais'] = yhat_reais
df_output['y_reais'] = y_reais
df_output['errors'] = errors
df_output['area_util'] = np.exp(X_train[:,0])
df_output['iptu'] = np.exp(X_train[:,1])
df_output['condominio'] = np.exp(X_train[:,2])
df_output.plot.scatter('y_reais', 'errors')
plt.show()
df_output.plot.scatter('area_util', 'errors')
plt.show()
df_output.plot.scatter('iptu', 'errors')
plt.show()
df_output.plot.scatter('condominio', 'errors')
plt.show()
'''

# test the model
model = RandomForestRegressor(n_estimators = n_trees, criterion = 'mse', n_jobs = -1,  random_state = random_state)
model.fit(X_train, y_train)
yhat = model.predict(X_test)

# put estimated prices back in R$
yhat_reais = np.exp(yhat)

# put observed prices back in R$
y_reais = np.exp(y_test)

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

# estimate uncertainty
variances = fci.random_forest_error(model, X_train, X_test)
plt.errorbar(y_test, yhat, yerr = np.sqrt(variances), fmt = 'o', ecolor = 'red')
plt.plot([10, 16], [10, 16], 'k--')
plt.xlabel('actual price, in log(R$)')
plt.ylabel('predicted price, in log(R$)')
plt.show()

# check interval predictions
lower = yhat - np.sqrt(variances)
upper = yhat + np.sqrt(variances)
corrects = 0
for y_i, l, u in zip(y_test, lower, upper):
    if l <= y_i <= u:
        corrects += 1
print(corrects, 'corrects out of', len(yhat))

# price target apartment
yhats = []
for i in range(n_trees):
    yhat = model.estimators_[i].predict(x_new)
    yhats.append(np.exp(yhat))
pred_df = pd.DataFrame(yhats)
pred_df.columns = ['yhat']
lower = pred_df['yhat'].describe()['25%']
upper = pred_df['yhat'].describe()['75%']
mid = pred_df['yhat'].describe()['50%']
#print(pred_df['yhat'].describe())
#mean = pred_df['yhat'].describe()['mean']
#std = pred_df['yhat'].describe()['std']
#lower = mean - (1.96 * (std/np.sqrt(n_trees)))
#upper = mean + (1.96 * (std/np.sqrt(n_trees)))
print(lower, mid, upper)
