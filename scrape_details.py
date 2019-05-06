import time
import requests
import pandas as pd

# carrega arquivo CSV com a lista de anuncios (e respectivos URLs)
df = pd.read_csv('lista_wimoveis.csv')
hrefs = list(df['href'])
for i, href in enumerate(hrefs):
    if i < 2240:
        continue
    print(i)
    url = 'https://www.wimoveis.com.br' + href
    response = requests.get(url)
    if response.status_code == 200:
        # baixa detalhes do anuncio
        with open('details/anuncio{}.html'.format(i), mode = 'w') as f:
            f.write(response.text)
            time.sleep(2)
