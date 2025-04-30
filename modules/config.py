import os
from dotenv import load_dotenv
import logging

load_dotenv()

LOG_DIR = 'logs'
LOG_FILE = os.path.join(LOG_DIR, 'app.log')
DATA_DIR = 'data'
MODEL_DIR = 'model'

LATITUDE = -27.2
LONGITUDE = -49.6
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

# Aponta para os arquivos gerados pelo generate_fake_data.py
HISTORICAL_FLOODS_CSV = os.path.join(DATA_DIR, 'historical_floods_faked.csv')
HISTORICAL_WEATHER_CSV = os.path.join(DATA_DIR, 'historical_weather_faked.csv')
# Arquivo de saída da previsão principal
FORECAST_OUTPUT_CSV = os.path.join(DATA_DIR, 'flood_forecast_output.csv')
# Arquivo do modelo ML que será GERADO/LIDO
MODEL_FILE = os.path.join(MODEL_DIR, 'flood_predictor.pkl')

# --- Configurações do Web Scraping ---
RIVER_LEVEL_URL = "https://defesacivil.riodosul.sc.gov.br/index.php?r=externo%2Fmetragem"
SCRAPING_TIMEOUT = 15 # Segundos

# --- Configurações do Cálculo de Risco ---
PRECIPITATION_PERCENTILE_THRESHOLD = 0.80
NDWI_IMPACT_THRESHOLD = 0.3
NDWI_LEVEL_ADJUSTMENT = 0.15
NDWI_RISK_ADJUSTMENT = 1

# --- Configurações do Google Earth Engine ---
USE_NDWI = False # GEE/NDWI desabilitado

# --- Configurações do Modelo ML (sem NDWI) ---
ML_FEATURES = ['temperatura_media', 'umidade_media', 'precipitacao_total', 'nivel_rio_historico_ou_atual']
TARGET_VARIABLE = 'Enchente'

# --- Setup ---
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

if not OPENWEATHER_API_KEY:
    logging.warning("OPENWEATHER_API_KEY não encontrada no .env.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[ logging.FileHandler(LOG_FILE), logging.StreamHandler() ]
)