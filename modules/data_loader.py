import pandas as pd
import pickle
import logging
import os
from datetime import datetime

# Importa a configuração para usar os paths dos arquivos CSV e do modelo
from . import config

# Função auxiliar para tentar converter datas variadas
def parse_custom_date(year, date_str):
    """Tenta converter formatos de data variados para YYYY-MM-DD."""
    if pd.isna(date_str) or pd.isna(year): return None
    try: year = int(year)
    except ValueError: return None
    date_str = str(date_str).strip().lower()
    try:
        months_pt_br = {'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4, 'maio': 5, 'junho': 6,'julho': 7, 'agosto': 8, 'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12}
        if ' de ' in date_str:
            parts = date_str.split(' de ');
            if len(parts) == 2 and parts[0].isdigit():
                day = int(parts[0]); month_name = parts[1]; month = months_pt_br.get(month_name)
                if month:
                    try: return pd.Timestamp(year=year, month=month, day=day)
                    except ValueError: return None # Dia inválido para o mês
            return None # Formato "DD de Mes" inválido
        elif date_str in months_pt_br: # Formato "Mes"
            month = months_pt_br[date_str]; logging.debug(f"Data incompleta '{date_str} {year}'. Usando dia 15.")
            try: return pd.Timestamp(year=year, month=month, day=15) # Usa dia 15
            except ValueError: return None
        else: # Tenta outros formatos
            parsed_date = pd.to_datetime(date_str, errors='coerce')
            if pd.notna(parsed_date):
                # Se conseguiu converter, mas o ano está errado ou é default (1900), ajusta o ano
                if parsed_date.year != year:
                    try: return pd.Timestamp(year=year, month=parsed_date.month, day=parsed_date.day)
                    except ValueError: return None
                else: return parsed_date # Ano já está correto
            else: # Tenta concatenar AAAA com o que foi lido (ex: MM-DD)
                try: return pd.to_datetime(f"{year}-{date_str}", errors='coerce')
                except ValueError: return None
    except Exception as e:
        logging.debug(f"Erro inesperado ao converter data: Ano={year}, Data='{date_str}'. Erro: {e}")
        return None

def load_historical_flood_data(file_path=config.HISTORICAL_FLOODS_CSV):
    """
    Carrega dados históricos (do arquivo _faked.csv que contém Enchente=0 e 1).
    Processa datas e garante a existência da coluna Data_Dia.
    """
    if not os.path.exists(file_path):
        logging.error(f"Arquivo de dados históricos não encontrado: {file_path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}. Execute generate_fake_data.py?")
    try:
        data = pd.read_csv(file_path, dtype={'Ano': 'Int64'})
        logging.info(f"Carregado {file_path}, {len(data)} registros.")

        required_cols = ['Ano', 'Data do pico da cheia', 'Metragem', 'Enchente']
        if not all(col in data.columns for col in required_cols):
             missing = [c for c in required_cols if c not in data.columns]
             logging.error(f"Colunas necessárias ({missing}) não encontradas em {file_path}")
             return None

        # --- Processar Datas ---
        # Tenta converter a coluna 'Data do pico da cheia' (que deve estar como AAAA-MM-DD)
        logging.info("Formatando coluna 'Data do pico da cheia' para datetime...")
        data['Data_Pico_Formatada'] = pd.to_datetime(data['Data do pico da cheia'], errors='coerce')

        # Verificar e remover linhas onde a data ou Metragem ou Enchente são inválidas/ausentes
        original_rows = len(data)
        required_for_ml = ['Data_Pico_Formatada', 'Metragem', 'Enchente']
        data = data.dropna(subset=required_for_ml)
        if len(data) < original_rows:
            logging.warning(f"{original_rows - len(data)} linhas removidas devido a valores ausentes em {required_for_ml}.")

        if data.empty:
            logging.error("Nenhum registro válido encontrado após limpeza.")
            return None

        data['Data_Dia'] = data['Data_Pico_Formatada'].dt.date

        # --- Tratar Ausentes Opcionais ---
        if 'Volume mm' in data.columns:
            mean_volume = data['Volume mm'].mean()
            data['Volume mm'] = data['Volume mm'].fillna(mean_volume)
        if 'Dias de chuva' in data.columns:
            mean_days = data['Dias de chuva'].mean()
            data['Dias de chuva'] = data['Dias de chuva'].fillna(mean_days)

        logging.info(f"Processamento de {file_path} concluído. {len(data)} registros válidos retidos.")
        return data

    except Exception as e:
        logging.error(f"Erro ao carregar ou processar {file_path}: {e}")
        return None

# --- Função para carregar Historical Weather (Fictício) ---
def load_historical_weather_data(file_path=config.HISTORICAL_WEATHER_CSV):
    """ Carrega dados meteorológicos históricos (fictícios) do CSV. """
    if not os.path.exists(file_path):
        logging.error(f"Arquivo de dados meteorológicos históricos não encontrado: {file_path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}. Execute generate_fake_data.py?")
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Carregado {file_path}, {len(data)} registros.")
        required_cols = ['timestamp', 'temperature', 'humidity', 'precipitation']
        if not all(col in data.columns for col in required_cols):
             missing = [c for c in required_cols if c not in data.columns]
             raise ValueError(f"Colunas necessárias ({missing}) não encontradas em {file_path}")

        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        data = data.dropna(subset=['timestamp']) # Remove linhas onde timestamp falhou

        if data.empty:
            logging.error(f"Nenhum timestamp válido encontrado em {file_path}.")
            return None

        data['Data_Dia'] = data['timestamp'].dt.date # Cria Data_Dia para merge

        # Trata NaNs remanescentes em colunas numéricas
        for col in ['temperature', 'humidity', 'precipitation']:
            # Verifica se coluna existe antes de tentar preencher
            if col in data.columns and data[col].isnull().any():
                mean_val = data[col].mean()
                if pd.notna(mean_val): # Só preenche se a média for válida
                    data[col] = data[col].fillna(mean_val)
                else: # Se a média for NaN (coluna toda NaN), preenche com 0
                    data[col] = data[col].fillna(0)
                logging.debug(f"Valores ausentes em '{col}' preenchidos.")

        return data
    except Exception as e:
        logging.error(f"Erro ao carregar ou processar {file_path}: {e}")
        return None

# --- Função para carregar Modelo ML ---
def load_ml_model(file_path=config.MODEL_FILE):
    """ Carrega o modelo de Machine Learning treinado (.pkl). """
    if not os.path.exists(file_path):
        logging.error(f"Arquivo do modelo ML não encontrado: {file_path}. Execute ml_trainer.py.")
        return None
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Modelo ML carregado de {file_path}.")
        return model
    except Exception as e:
        logging.error(f"Erro inesperado ao carregar o modelo ML: {e}")
        return None