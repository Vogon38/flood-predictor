import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import logging
import os

from modules import config, data_loader

# --- Configuração ---
logging.info("--- Iniciando Treinamento do Modelo ML (com Dados Fictícios) ---")
logging.warning("!!! ATENÇÃO: Modelo será treinado com dados HISTÓRICOS FICTÍCIOS. Não confiável para uso real. !!!")

# --- 1. Carregar Dados (Fictícios) ---
try:
    logging.info(f"Carregando dados históricos de enchentes/não-enchentes de: {config.HISTORICAL_FLOODS_CSV}")
    historical_data = data_loader.load_historical_flood_data(config.HISTORICAL_FLOODS_CSV)

    logging.info(f"Carregando dados meteorológicos fictícios de: {config.HISTORICAL_WEATHER_CSV}")
    weather_data = data_loader.load_historical_weather_data(config.HISTORICAL_WEATHER_CSV)

    if historical_data is None or weather_data is None:
        logging.error("Falha ao carregar dados históricos e/ou meteorológicos fictícios. Verifique os arquivos em /data e os loaders. Abortando.")
        exit()
    if historical_data.empty or weather_data.empty:
         logging.error("Um ou ambos os dataframes estão vazios após o carregamento. Abortando.")
         exit()

except FileNotFoundError as e:
     logging.error(f"Erro: {e}")
     logging.error("Certifique-se de que o script generate_fake_data.py foi executado com sucesso.")
     exit()
except Exception as e:
     logging.error(f"Erro inesperado ao carregar dados: {e}")
     exit()


# --- 2. Preparar Dados para Treinamento ---
logging.info("Mesclando dados históricos e meteorológicos fictícios...")

# Certificar que as colunas de data/dia existem
if 'Data_Dia' not in historical_data.columns or 'Data_Dia' not in weather_data.columns:
    logging.error("Coluna 'Data_Dia' não encontrada nos dataframes carregados após loaders. Verifique data_loader.py.")
    exit()

# Agrupar dados de clima por dia (pega a primeira ocorrência se houver duplicatas)
weather_daily_agg = weather_data.drop_duplicates(subset=['Data_Dia']).copy()

# Renomear Metragem antes do merge
if 'Metragem' in historical_data.columns:
     historical_data.rename(columns={'Metragem': 'nivel_rio_historico_ou_atual'}, inplace=True)
elif 'nivel_rio_historico_ou_atual' not in historical_data.columns:
     logging.error("Coluna 'Metragem' (ou 'nivel_rio_historico_ou_atual') não encontrada nos dados históricos.")
     exit()

# Selecionar colunas necessárias do historical_data
cols_from_hist = ['Data_Dia', 'nivel_rio_historico_ou_atual', config.TARGET_VARIABLE]
# Verifica se todas as colunas existem antes de selecionar
cols_hist_exist = [col for col in cols_from_hist if col in historical_data.columns]
if len(cols_hist_exist) != len(cols_from_hist):
     missing = list(set(cols_from_hist) - set(cols_hist_exist))
     logging.error(f"Colunas faltando em historical_data para o merge: {missing}")
     exit()

# Selecionar colunas necessárias do weather_data
cols_from_weather = ['Data_Dia', 'temperature', 'humidity', 'precipitation']
cols_weather_exist = [col for col in cols_from_weather if col in weather_daily_agg.columns]
if len(cols_weather_exist) != len(cols_from_weather):
     missing = list(set(cols_from_weather) - set(cols_weather_exist))
     logging.error(f"Colunas faltando em weather_data para o merge: {missing}")
     exit()

# Realizar o merge
merged_data = pd.merge(
    historical_data[cols_hist_exist],
    weather_daily_agg[cols_weather_exist],
    on='Data_Dia',
    how='inner'
)

# Renomear colunas do clima para corresponder às features esperadas
merged_data.rename(columns={
    'temperature': 'temperatura_media',
    'humidity': 'umidade_media',
    'precipitation': 'precipitacao_total'
}, inplace=True)

logging.info(f"Dados mesclados para treinamento: {len(merged_data)} registros.")
if merged_data.empty:
    logging.error("Merge resultou em dataframe vazio. Verifique as datas e o processo.")
    exit()

# --- 3. Adicionar Feature NDWI (Removido) ---
# if config.USE_NDWI: ...

# --- 4. Selecionar Features e Target ---
logging.info(f"Selecionando features: {config.ML_FEATURES}")
logging.info(f"Variável alvo: {config.TARGET_VARIABLE}")

missing_features = [f for f in config.ML_FEATURES if f not in merged_data.columns]
if missing_features:
    logging.error(f"Features configuradas ({config.ML_FEATURES}) não encontradas nos dados mesclados: {missing_features}. Colunas disponíveis: {merged_data.columns.tolist()}")
    exit()

X = merged_data[config.ML_FEATURES]
y = merged_data[config.TARGET_VARIABLE]

# Verificar balanceamento
class_counts = y.value_counts(normalize=True)
logging.info(f"Distribuição das classes (0=Não Enchente, 1=Enchente):\n{class_counts}")
if len(class_counts) < 2:
     logging.error("Os dados de treinamento contêm apenas uma classe. Verifique a geração de dados fictícios.")
     exit()

# Verificar NaNs nas features X
if X.isnull().values.any():
    logging.warning("Valores ausentes (NaN) encontrados nas features X. Preenchendo com a média da coluna.")
    X = X.fillna(X.mean())
    if X.isnull().values.any():
        logging.error("Ainda existem NaNs nas features após preenchimento com média. Verifique os dados.")
        exit()

# --- 5. Dividir Dados em Treino e Teste ---
logging.info("Dividindo dados em conjuntos de treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
logging.info(f"Tamanho Treino: {len(X_train)} | Tamanho Teste: {len(X_test)}")

# --- 6. Treinar o Modelo (Random Forest) ---
logging.info("Treinando o modelo RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced'
)
model.fit(X_train, y_train)

# --- 7. Avaliar o Modelo ---
logging.info("Avaliando o modelo no conjunto de teste...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Acurácia (com dados fictícios): {accuracy:.4f}")
logging.info("Relatório de Classificação (com dados fictícios):")
# Adicionado zero_division=0 para evitar warning se uma classe não tiver predições/suporte
print(classification_report(y_test, y_pred, target_names=['Não Enchente (0)', 'Enchente (1)'], zero_division=0))
logging.info("Matriz de Confusão (com dados fictícios):")
print(confusion_matrix(y_test, y_pred))

# --- 8. Salvar o Modelo Treinado ---
model_path = config.MODEL_FILE
logging.info(f"Salvando o modelo treinado em: {model_path}")
try:
    # Garante que o diretório existe
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info("Modelo salvo com sucesso!")
except Exception as e:
    logging.error(f"Erro ao salvar o modelo: {e}")

logging.info("--- Treinamento Concluído ---")