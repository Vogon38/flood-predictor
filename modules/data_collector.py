import requests
from bs4 import BeautifulSoup
import logging
import numpy as np
from datetime import datetime, timedelta
import ee
import ee.mapclient

# Importar configurações
from . import config

# --- Flag para controlar inicialização do GEE ---
_gee_initialized = False

def initialize_gee():
    """Inicializa a API do Google Earth Engine com ID do projeto."""
    global _gee_initialized
    if _gee_initialized:
        return True
    try:
        project_id = config.GEE_PROJECT_ID
        if not project_id:
            logging.warning("GEE_PROJECT_ID não definido. Tentando ee.Initialize() sem projeto...")
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        else:
            logging.info(f"Tentando inicializar Google Earth Engine com projeto: {project_id}...")
            ee.Initialize(project=project_id, opt_url='https://earthengine-highvolume.googleapis.com')

        logging.info("Google Earth Engine inicializado com sucesso.")
        _gee_initialized = True
        return True
    except ee.EEException as e:
        logging.warning(f"Falha na inicialização GEE (mesmo com/sem projeto): {e}. Tentando autenticar...")
        try:
            ee.Authenticate()
            project_id = config.GEE_PROJECT_ID
            if not project_id:
                 logging.warning("GEE_PROJECT_ID não definido. Tentando ee.Initialize() sem projeto pós-auth...")
                 ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
            else:
                logging.info(f"Tentando inicializar Google Earth Engine com projeto {project_id} pós-auth...")
                ee.Initialize(project=project_id, opt_url='https://earthengine-highvolume.googleapis.com')

            logging.info("Google Earth Engine autenticado e inicializado com sucesso.")
            _gee_initialized = True
            return True
        except Exception as auth_e:
            logging.error(f"Falha crítica ao autenticar/inicializar Google Earth Engine: {auth_e}")
            _gee_initialized = False
            return False

# --- Coleta de Previsão do Tempo (OpenWeatherMap) ---
def get_weather_forecast(lat, lon, api_key):
    """
    Busca a previsão do tempo 5 dias/3 horas da API OpenWeatherMap
    e agrega os dados por dia.
    """
    if not api_key:
        logging.error("API Key do OpenWeatherMap não fornecida.")
        return None

    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={api_key}&lang=pt_br"
    processed_data = {}
    try:
        response = requests.get(url, timeout=config.SCRAPING_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        daily_data = {}
        for forecast in data.get('list', []):
            dt_txt = forecast.get('dt_txt', '')
            if not dt_txt: continue
            date = dt_txt.split(' ')[0]
            rain = forecast.get('rain', {}).get('3h', 0.0)
            temp = forecast.get('main', {}).get('temp')
            humidity = forecast.get('main', {}).get('humidity')

            if temp is None or humidity is None: continue

            if date not in daily_data:
                daily_data[date] = {'precipitation_total': 0.0, 'temperatures': [], 'humidities': [], 'descriptions': set()}

            daily_data[date]['precipitation_total'] += float(rain)
            daily_data[date]['temperatures'].append(float(temp))
            daily_data[date]['humidities'].append(float(humidity))
            weather_desc = forecast.get('weather', [{}])[0].get('description')
            if weather_desc: daily_data[date]['descriptions'].add(weather_desc)

        for date, values in daily_data.items():
            avg_temp = np.mean(values['temperatures']) if values['temperatures'] else None
            avg_humidity = np.mean(values['humidities']) if values['humidities'] else None
            desc = ', '.join(sorted(list(values['descriptions'])))

            if avg_temp is not None and avg_humidity is not None:
                 processed_data[date] = {
                    'temperatura_media': round(avg_temp, 2),
                    'umidade_media': round(avg_humidity, 2),
                    'precipitacao_total': round(values['precipitation_total'], 2),
                    'descricao': desc
                }

        logging.info(f"Previsão do tempo obtida para {len(processed_data)} dias.")
        return processed_data

    except requests.exceptions.RequestException as e:
        logging.error(f"Erro na requisição da previsão do tempo: {e}")
        return None
    except Exception as e:
        logging.error(f"Erro inesperado ao processar previsão do tempo: {e}")
        return None

# --- Coleta de Nível do Rio (Web Scraping) ---
def get_current_river_level():
    """
    Busca o nível atual do rio via web scraping do site da Defesa Civil.
    Versão robustecida com melhor tratamento de erros.
    """
    url = config.RIVER_LEVEL_URL
    logging.info(f"Tentando obter nível do rio de: {url}")
    try:
        response = requests.get(url, timeout=config.SCRAPING_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        river_level_row = soup.find('tr', {'data-key': '0'})
        if river_level_row:
            cells = river_level_row.find_all('td')
            if len(cells) > 1:
                level_text = cells[1].get_text(strip=True)
                try:
                    level = float(level_text.replace('m', '').replace(',', '.').strip())
                    logging.info(f"Nível do rio obtido (data-key='0'): {level:.2f} m")
                    return level
                except (ValueError, IndexError):
                    logging.warning(f"Não foi possível converter/extrair texto '{level_text}' (de data-key='0').")
        return None

    except requests.exceptions.Timeout:
        logging.error(f"Timeout ao tentar acessar {url}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro na requisição HTTP para obter nível do rio: {e}")
        return None
    except Exception as e:
        logging.error(f"Erro inesperado durante o scraping do nível do rio: {e}")
        return None

# --- Função de Máscara de Nuvem para Landsat SR ---
def mask_landsat_sr_clouds(image):
    """Aplica máscara de nuvens/sombras em imagens Landsat 8/9 SR (Collection 2)."""
    qa_pixel = image.select('QA_PIXEL')
    cloud_shadow_bit_mask = 1 << 3
    snow_bit_mask = 1 << 4
    cloud_bit_mask = 1 << 5
    clear = qa_pixel.bitwiseAnd(cloud_shadow_bit_mask).eq(0) \
                   .And(qa_pixel.bitwiseAnd(snow_bit_mask).eq(0)) \
                   .And(qa_pixel.bitwiseAnd(cloud_bit_mask).eq(0))
    return image.updateMask(clear)

# --- Função de Cálculo de NDWI para Imagem GEE ---
def calculate_gee_ndwi(image):
    """Calcula NDWI para uma imagem Landsat SR GEE."""
    scale = 0.0000275
    offset = -0.2
    green = image.select('SR_B3').multiply(scale).add(offset)
    nir = image.select('SR_B5').multiply(scale).add(offset)
    ndwi = green.subtract(nir).divide(green.add(nir).add(1e-9)).rename('NDWI')
    return image.addBands(ndwi)

# --- Coleta de NDWI via Google Earth Engine ---
def get_current_ndwi_gee(aoi_coords, start_date, end_date, max_cloud_percent):
    """
    Busca a imagem Landsat SR mais recente e menos nublada para a AOI
    e calcula o NDWI médio.
    """
    if not _gee_initialized:
        logging.error("GEE não inicializado. Não é possível buscar NDWI.")
        return None

    try:
        aoi = ee.Geometry.Polygon(aoi_coords)
        logging.info(f"Buscando imagem Landsat SR para AOI entre {start_date} e {end_date}")

        l8_sr = ee.ImageCollection(config.LANDSAT_COLLECTION_SR) \
                    .filterBounds(aoi) \
                    .filterDate(start_date, end_date) \
                    .map(mask_landsat_sr_clouds)
        l9_sr = ee.ImageCollection(config.LANDSAT9_COLLECTION_SR) \
                    .filterBounds(aoi) \
                    .filterDate(start_date, end_date) \
                    .map(mask_landsat_sr_clouds)
        merged_collection = l8_sr.merge(l9_sr)
        filtered_collection = merged_collection.filter(ee.Filter.lt('CLOUD_COVER', max_cloud_percent))
        best_image = filtered_collection.sort('CLOUD_COVER').first()

        if best_image is None:
            logging.warning(f"Nenhuma imagem Landsat SR encontrada com < {max_cloud_percent}% nuvens para o período {start_date} a {end_date} e AOI.")
            return None

        image_with_ndwi = calculate_gee_ndwi(best_image)
        ndwi_stats = image_with_ndwi.select('NDWI').reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=100, maxPixels=1e9
        )
        mean_ndwi = ndwi_stats.get('NDWI')
        mean_ndwi_value = mean_ndwi.getInfo()

        if mean_ndwi_value is None:
             logging.warning("Não foi possível calcular o NDWI médio (pode ser falta de pixels válidos na AOI).")
             return None
        else:
            image_date = ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            cloud_cover = best_image.get('CLOUD_COVER').getInfo()
            logging.info(f"NDWI médio calculado via GEE: {mean_ndwi_value:.4f} (Imagem de {image_date}, Nuvem: {cloud_cover:.1f}%)")
            return mean_ndwi_value

    except ee.EEException as e:
        logging.error(f"Erro durante a execução no Google Earth Engine: {e}")
        return None
    except Exception as e:
        logging.error(f"Erro inesperado ao buscar NDWI via GEE: {e}")
        return None

# --- Função Principal de Coleta de NDWI (agora usa GEE) ---
def get_ndwi_value():
    """
    Obtém o valor médio de NDWI mais recente para a AOI usando GEE.
    """
    if not config.USE_NDWI:
        logging.info("Coleta de NDWI desabilitada na configuração.")
        return None
    if not initialize_gee():
         logging.error("Falha ao inicializar GEE. Coleta de NDWI abortada.")
         return None

    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.MAX_PAST_DAYS_FOR_SATELLITE_IMAGE)
    ndwi_mean = get_current_ndwi_gee(
        aoi_coords=config.GEE_AOI_COORDS,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        max_cloud_percent=config.MAX_CLOUD_COVER_PERCENT
    )
    return ndwi_mean

# --- Final do Arquivo ---