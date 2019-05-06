import time
import requests

base_url = 'https://www.wimoveis.com.br/'

for i in range(1, 253): # numero de paginas de resultados no Wimoveis
    print('pagina', i)
    query_url = base_url + 'comerciais-venda-distrito-federal-goias-pagina-{}.html'.format(i)
    response = requests.get(query_url)
    if response.status_code == 200:
        # salva codigo-fonte do anuncio
        with open('list/pagina{}.html'.format(i), mode = 'w') as f:
            f.write(response.text)
            time.sleep(2)
