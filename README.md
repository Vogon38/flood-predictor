# Previsão de Enchente (Rio do Sul - Exemplo Acadêmico)

## Objetivo

Este projeto tem como objetivo demonstrar a criação de um sistema de previsão de risco de enchentes, utilizando a cidade de Rio do Sul (SC) como estudo de caso. Ele combina duas abordagens para gerar alertas:

1.  **Cálculo de Risco por Limiares:** Compara o nível atual do rio e uma estimativa futura (baseada na previsão de chuva) com níveis históricos de enchentes para gerar um score de risco (1-5).  
2.  **Modelo de Machine Learning (Random Forest):** Classifica o risco futuro como "Sim" ou "Não" para enchente, com base em dados históricos e previsão do tempo. **Importante:** Para fins de demonstração e devido à dificuldade de obter dados históricos climáticos completos e gratuitos, este modelo é treinado usando **dados meteorológicos históricos fictícios** e **exemplos fictícios de dias sem enchente**.

**AVISO FUNDAMENTAL:** O modelo de Machine Learning incluído neste projeto é treinado com **DADOS FICTÍCIOS**. Suas previsões servem apenas como **demonstração acadêmica** do processo e **NÃO SÃO CONFIÁVEIS** para a tomada de decisões reais sobre segurança ou evacuação. A previsão mais realista (embora ainda simplificada) é a gerada pelo método de **Cálculo de Risco por Limiares**, que utiliza dados reais de nível histórico, nível atual e previsão do tempo.

## Dados Utilizados

* **Dados Reais:**
    * Níveis máximos de enchentes históricas (`Metragem` do `historical_floods.csv` original) - Usados para calcular limites estatísticos.
    * Nível atual do rio (obtido via web scraping em tempo real).
    * Previsão do tempo para os próximos dias (Temperatura, Umidade, Precipitação via API OpenWeatherMap).
* **Dados Fictícios (Gerados para o Modelo ML):**
    * Exemplos de dias sem enchente (`Enchente = 0`): Datas e níveis baixos de rio gerados artificialmente.
    * Clima histórico (`historical_weather_faked.csv`): Valores de temperatura, umidade e precipitação gerados artificialmente para *todas* as datas (enchentes reais e não-enchentes fictícias), tentando simular condições mais chuvosas para dias de enchente e mais secas para dias normais.

## Estrutura do Projeto
rio_do_sul_flood_predictor/  
│  
├── data/                       
│   ├── historical_floods.csv       
│   ├── historical_floods_faked.csv   
│   ├── historical_weather_faked.csv   
│   └── flood_forecast_output.csv   
│   └── flood_risk_comparison.png   
│  
├── logs/                       
│   └── app.log  
│  
├── model/                      
│   └── flood_predictor.pkl     
│  
├── modules/                    
│   ├── init.py  
│   ├── config.py               
│   ├── data_collector.py       
│   ├── data_loader.py          
│   ├── risk_calculator.py      
│   ├── ml_predictor.py         
│   └── plotter.py              
│  
├── .env                      # Chaves de API (NÃO versionar)  
├── requirements.txt            
├── main.py                     
├── ml_trainer.py               
└── generate_fake_data.py       
└── .gitignore                  


## Como Funciona (Fluxo Principal)

1.  **Geração de Dados Fictícios (Execução Única):** O script `generate_fake_data.py` lê o `historical_floods.csv` original, gera exemplos de dias sem enchente, combina tudo e gera um arquivo de clima fictício correspondente. Salva `historical_floods_faked.csv` e `historical_weather_faked.csv`.
2.  **Treinamento do Modelo ML (Execução Única):** O script `ml_trainer.py` usa os dois arquivos `_faked.csv` para treinar o modelo RandomForest e salva o `model/flood_predictor.pkl`.
3.  **Execução da Previsão (`main.py`):**
    * Carrega os dados históricos de `historical_floods_faked.csv` para calcular limites estatísticos.
    * Busca o nível atual do rio via scraping.
    * Busca a previsão do tempo (OpenWeatherMap).
    * Calcula o risco baseado em limiares para os próximos dias.
    * Carrega o modelo `flood_predictor.pkl`.
    * Faz a previsão ML (Sim/Não) para os próximos dias.
    * Salva os resultados combinados e gera o gráfico.

## Instruções de Instalação e Configuração

1.  **Clone o Repositório:**
    ```bash
    git clone <url-do-seu-repositorio>
    cd rio_do_sul_flood_predictor
    ```
2.  **Crie e Ative um Ambiente Virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS ou venv\Scripts\activate no Windows
    ```
3.  **Instale as Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure a Chave da API:**
    * Crie o arquivo `.env` na raiz do projeto.
    * Adicione sua chave da API do [OpenWeatherMap](https://openweathermap.org/):
        ```
        OPENWEATHER_API_KEY=SUA_CHAVE_VALIDA_AQUI
        ```
5.  **Revise `modules/config.py`:**
    * Verifique se `LATITUDE`, `LONGITUDE` estão corretas para Rio do Sul ou outra cidade qualquer que você queira usar.
    * Confirme se os paths em `HISTORICAL_FLOODS_CSV` e `HISTORICAL_WEATHER_CSV` apontam para os arquivos `_faked.csv`.
    * Verifique se `USE_NDWI` está como `False` (pois desabilitei GEE).

## Preparação dos Dados (Essencial!)

1.  **Arquivo Histórico Original:**
    * Coloque seu arquivo CSV contendo os registros **reais** de picos de enchente na pasta `data/` com o nome **`historical_floods.csv`**.
    * **IMPORTANTE:** Este arquivo precisa ter as colunas `Ano`, `Data do pico da cheia` e `Metragem`. Tente deixar a coluna de data o mais completa possível (AAAA-MM-DD é ideal, mas o script `data_loader` tenta lidar com outros formatos). O script `generate_fake_data.py` lerá este arquivo.
2.  **Gere os Dados Fictícios:**
    * Execute o script para criar os arquivos necessários para o treinamento do ML:
        ```bash
        python generate_fake_data.py
        ```
    * Isso criará os arquivos `data/historical_floods_faked.csv` e `data/historical_weather_faked.csv`.

## Execução do Projeto

Após a configuração e preparação dos dados, siga esta ordem:

1.  **Treine o Modelo ML (Executar uma vez após gerar dados):**
    ```bash
    python ml_trainer.py
    ```
    Isso criará o arquivo `model/flood_predictor.pkl`.

2.  **Execute a Previsão Principal (Executar para obter previsões):**
    ```bash
    python main.py
    ```
    Isso buscará dados atuais, calculará os riscos (limiar e ML fictício), salvará o CSV e gerará o gráfico.

## Saídas

* **`data/flood_forecast_output.csv`**: Contém a previsão diária detalhada, incluindo:
    * Dados da previsão do tempo (temp, umid, precip).
    * Nível atual base.
    * Nível estimado pelo método de limiares.
    * Score de risco (1-5) e categoria pelo método de limiares.
    * Previsão ML (Sim/Não) com probabilidade (baseada em dados fictícios).
* **`data/flood_risk_comparison.png`**: Gráfico visual comparando nível atual, estimado (limiar) e distribuição histórica.
* **Logs (`logs/app.log` e Console)**: Informações detalhadas da execução.

## Como Adaptar para Outra Região

Adaptar este projeto para outra cidade ou região exigirá modificações significativas:

1.  **Configuração (`modules/config.py`):**
    * Altere `LATITUDE` e `LONGITUDE` para as coordenadas da nova região.
    * **MUITO IMPORTANTE:** Altere `RIVER_LEVEL_URL` para a URL da página web que contém o nível do rio da *nova região*. Se não houver uma página similar, a função de web scraping precisará ser completamente reescrita ou substituída por outra fonte de dados.
2.  **Dados Históricos (`data/historical_floods.csv`):**
    * **Substitua** o arquivo `historical_floods.csv` pelos dados de enchentes da *nova região*. O formato deve ser similar (com `Ano`, `Data do pico da cheia`, `Metragem`).
    * **Idealmente:** Adicione dados reais de dias **sem enchente** (`Enchente=0`) para a nova região, se disponíveis. Isso melhoraria muito o modelo ML se você decidir usar dados reais no futuro.
3.  **Web Scraping (`modules/data_collector.py`):**
    * A função `get_current_river_level` **precisará ser reescrita** para se adequar à estrutura HTML da nova fonte de dados do nível do rio. Cada site é diferente, então esta é uma parte que exige análise e codificação específica para o novo site.
4.  **Dados Fictícios (`generate_fake_data.py`):**
    * Ajuste as faixas de valores (temperatura, umidade, precipitação, nível normal do rio) para serem realistas para a *nova região*.
    * Execute-o novamente para gerar os arquivos `_faked.csv` específicos para a nova região (baseados no novo `historical_floods.csv`).
5.  **Treinamento (`ml_trainer.py`):**
    * Execute-o novamente após gerar os novos dados fictícios para criar um modelo `.pkl` adaptado (ainda que ficticiamente) à nova região.
6.  **Calibração (`modules/risk_calculator.py`):**
    * Os limites históricos (média, DP) serão recalculados automaticamente a partir dos novos dados.
    * No entanto, a **fórmula** que estima o `aumento_estimado` do nível do rio baseado na chuva pode precisar de **calibração** para a hidrologia da nova bacia. Isso pode exigir análise de dados reais da nova região, se disponíveis.
7.  **(Opcional) NDWI/GEE:**
    * Se quiser usar NDWI, altere `GEE_AOI_COORDS` em `config.py` para a nova área.
    * Defina `USE_NDWI = True`.
    * Instale as dependências GEE e passe pelo processo de configuração/autenticação do GEE.

A adaptação mais desafiadora geralmente é obter os dados históricos e ajustar a coleta de dados em tempo real (web scraping) para a nova localidade.

## Próximos Passos e Melhorias (Sugestões)

* **Dados Reais para ML:** A melhoria mais significativa seria substituir os dados fictícios por dados reais:
    * **Não-Enchente:** Coletar datas e níveis de rio reais para dias *sem* enchente na região.
    * **Clima Histórico:** Obter séries históricas reais de temperatura, umidade e precipitação diária para a região. Isso permitiria treinar um modelo ML com potencial de previsão real.
* **Integração Google Earth Engine (GEE):**
    * Ativar a opção `USE_NDWI = True` em `config.py` (requer configuração de projeto Cloud, habilitação de API e autenticação GEE).
    * Usar o NDWI atual como feature adicional no cálculo de risco ou na predição ML.
    * (Avançado) Usar GEE para buscar NDWI *histórico* para as datas de treino, substituindo a necessidade de gerar clima fictício ou complementando dados reais.
* **Calibração do Modelo de Limiares:** Validar e refinar a fórmula em `risk_calculator.py` que estima o aumento do nível do rio com base na chuva, usando dados observados de chuva x aumento do nível para a bacia.
* **Outras Variáveis:** Incluir dados de estações a montante (rio acima), níveis de reservatórios (barragens), ou outras variáveis relevantes.
* **Interface/Visualização:** Criar um dashboard web para apresentar as previsões de forma mais interativa.
