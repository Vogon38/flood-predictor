import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import logging
import os

from modules import config
from modules.data_loader import load_historical_flood_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Parâmetros da Geração Fictícia ---
NUM_NON_FLOOD_EXAMPLES = 150
TEMP_RANGE_SUMMER = (18, 32); TEMP_RANGE_AUTUMN = (14, 28); TEMP_RANGE_WINTER = (8, 22); TEMP_RANGE_SPRING = (12, 26)
HUMIDITY_RANGE_NORMAL = (60, 90); HUMIDITY_RANGE_FLOOD = (75, 98)
PRECIP_RANGE_NORMAL = (0, 15); PRECIP_RANGE_FLOOD_RANDOM = (40, 180)
RIVER_LEVEL_NORMAL_RANGE = (0.5, 4.0)
VOLUME_MM_NORMAL = (0, 50); DIAS_CHUVA_NORMAL = (0, 3)

OUTPUT_FLOOD_FILE = config.HISTORICAL_FLOODS_CSV
OUTPUT_WEATHER_FILE = config.HISTORICAL_WEATHER_CSV
INPUT_FLOOD_FILE_ORIGINAL = config.ORIGINAL_HISTORICAL_FLOODS_CSV
# -----------------------------------------

def get_seasonal_temp_range(month):
    if month in [12, 1, 2]: return TEMP_RANGE_SUMMER
    elif month in [3, 4, 5]: return TEMP_RANGE_AUTUMN
    elif month in [6, 7, 8]: return TEMP_RANGE_WINTER
    elif month in [9, 10, 11]: return TEMP_RANGE_SPRING
    else: return (15, 25)

def generate_fake_data():
    logging.info("--- Iniciando Geração de Dados Fictícios (Completa) ---")

    logging.info(f"Carregando dados originais de: {INPUT_FLOOD_FILE_ORIGINAL}")
    df_floods_orig = load_historical_flood_data(INPUT_FLOOD_FILE_ORIGINAL)

    if df_floods_orig is None or df_floods_orig.empty:
        logging.error(f"Não foi possível carregar dados históricos originais de {INPUT_FLOOD_FILE_ORIGINAL}. Abortando.")
        return

    # Manter apenas registros com data válida e Metragem
    df_floods = df_floods_orig.dropna(subset=['Data_Pico_Formatada', 'Metragem']).copy()
    if df_floods.empty:
        logging.error("Nenhum registro com data válida e Metragem encontrado nos dados originais. Abortando.")
        return
    df_floods['Enchente'] = 1
    logging.info(f"{len(df_floods)} registros de enchente com data válida e Metragem retidos.")

    # --- Gerar Exemplos de Não Enchente (Enchente = 0) ---
    logging.info(f"Gerando {NUM_NON_FLOOD_EXAMPLES} exemplos fictícios de não-enchente...")
    non_flood_data = []
    existing_dates = set(df_floods['Data_Pico_Formatada'])
    min_year = df_floods['Ano'].min(); max_year = df_floods['Ano'].max()
    # Lida com possíveis NaNs em min/max_year se 'Ano' não for Int64 ou tiver nulos
    if pd.isna(min_year) or pd.isna(max_year): min_year, max_year = 1980, 2024 # Defaults

    attempts = 0
    while len(non_flood_data) < NUM_NON_FLOOD_EXAMPLES and attempts < NUM_NON_FLOOD_EXAMPLES * 5:
        attempts += 1
        try:
            year = random.randint(int(min_year), int(max_year))
            start_date = datetime(year, 1, 1); end_date = datetime(year, 12, 31)
            random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
            random_date_ts = pd.Timestamp(random_date.date())
            if random_date_ts not in existing_dates:
                metragem = round(random.uniform(RIVER_LEVEL_NORMAL_RANGE[0], RIVER_LEVEL_NORMAL_RANGE[1]), 2)
                volume = round(random.uniform(VOLUME_MM_NORMAL[0], VOLUME_MM_NORMAL[1]), 1)
                dias_chuva = random.randint(DIAS_CHUVA_NORMAL[0], DIAS_CHUVA_NORMAL[1])
                non_flood_data.append({
                    'Ano': year, 'Data do pico da cheia': random_date_ts.strftime('%Y-%m-%d'),
                    'Volume mm': volume, 'Metragem': metragem, 'Dias de chuva': dias_chuva,
                    'Enchente': 0, 'Data_Pico_Formatada': random_date_ts
                })
                existing_dates.add(random_date_ts)
        except ValueError: continue # Ignora datas inválidas (ex: 29 Fev não bissexto)

    df_non_floods = pd.DataFrame(non_flood_data)
    df_combined = pd.concat([df_floods, df_non_floods], ignore_index=True)

    # Salvar o arquivo de floods combinado (com Enchente=0 e Enchente=1)
    logging.info(f"Salvando dados combinados (reais+fictícios) em: {OUTPUT_FLOOD_FILE}")
    cols_to_save = ['Ano', 'Data do pico da cheia', 'Volume mm', 'Metragem', 'Dias de chuva', 'Enchente']
    # Garante que apenas colunas existentes sejam selecionadas antes de salvar
    cols_present = [col for col in cols_to_save if col in df_combined.columns]
    df_combined_to_save = df_combined[cols_present].copy()
    # Usa a Data_Pico_Formatada para garantir o formato correto antes de salvar
    if 'Data_Pico_Formatada' in df_combined.columns:
         df_combined_to_save['Data do pico da cheia'] = pd.to_datetime(df_combined['Data_Pico_Formatada']).dt.strftime('%Y-%m-%d')
    else: # Fallback se Data_Pico_Formatada não foi criada por algum motivo
         df_combined_to_save['Data do pico da cheia'] = pd.to_datetime(df_combined_to_save['Data do pico da cheia'], errors='coerce').dt.strftime('%Y-%m-%d')

    df_combined_to_save.to_csv(OUTPUT_FLOOD_FILE, index=False, float_format='%.2f')


    # --- Gerar Dados Meteorológicos Fictícios ---
    logging.info("Gerando dados meteorológicos fictícios (com sazonalidade e Volume mm)...")
    weather_data_list = []
    for index, row in df_combined.iterrows():
        date = row['Data_Pico_Formatada']
        is_flood = row['Enchente'] == 1
        if pd.isna(date): continue

        month = date.month
        temp_range = get_seasonal_temp_range(month)

        if is_flood:
            temp = round(random.uniform(temp_range[0], temp_range[1]) + random.uniform(0, 3), 1)
            humidity = round(random.uniform(HUMIDITY_RANGE_FLOOD[0], HUMIDITY_RANGE_FLOOD[1]), 1)
            precip = None
            if 'Volume mm' in row and pd.notna(row['Volume mm']):
                precip = round(float(row['Volume mm']), 1)
            else:
                precip = round(random.uniform(PRECIP_RANGE_FLOOD_RANDOM[0], PRECIP_RANGE_FLOOD_RANDOM[1]), 1)
            precip = max(0, precip)
        else:
            temp = round(random.uniform(temp_range[0], temp_range[1]), 1)
            humidity = round(random.uniform(HUMIDITY_RANGE_NORMAL[0], HUMIDITY_RANGE_NORMAL[1]), 1)
            precip = round(random.uniform(PRECIP_RANGE_NORMAL[0], PRECIP_RANGE_NORMAL[1]), 1)
            precip = max(0, precip)
            if precip < 5: humidity = max(HUMIDITY_RANGE_NORMAL[0], humidity - random.uniform(0,10))
            else: humidity = min(HUMIDITY_RANGE_NORMAL[1], humidity + random.uniform(0,10))

        weather_data_list.append({
            'timestamp': date.strftime('%Y-%m-%d 00:00:00'),
            'temperature': temp, 'humidity': round(humidity, 1), 'precipitation': precip
        })

    df_weather_fake = pd.DataFrame(weather_data_list)
    logging.info(f"Salvando dados meteorológicos fictícios em: {OUTPUT_WEATHER_FILE}")
    df_weather_fake.to_csv(OUTPUT_WEATHER_FILE, index=False, float_format='%.2f')

    logging.info("Geração de dados fictícios concluída!")


if __name__ == "__main__":
    # Garante que a pasta data existe
    os.makedirs(config.DATA_DIR, exist_ok=True)
    generate_fake_data()