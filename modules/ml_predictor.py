import pandas as pd
import logging
import numpy as np

from . import config
from .data_loader import load_ml_model

ml_model = None
try:
    ml_model = load_ml_model()
except Exception as e:
    logging.error(f"Erro inicial ao tentar carregar modelo ML: {e}")

def predict_ml_flood(daily_forecast_data, current_river_level):
    """
    Prevê o risco de enchente ('Sim'/'Não') usando o modelo ML treinado (com dados fictícios).
    Espera dados de previsão diária (temp, umidade, precip) e nível atual do rio.
    NÃO usa NDWI pois foi desabilitado.
    """
    global ml_model # Permite recarregar se falhou na primeira vez
    if ml_model is None:
         # Tenta recarregar caso tenha falhado na inicialização do módulo
         logging.warning("Modelo ML não carregado inicialmente. Tentando recarregar...")
         ml_model = load_ml_model()
         if ml_model is None:
             logging.error("Modelo ML não disponível. Não é possível fazer a predição ML.")
             return "ML Indisponível"

    try:
        # --- Preparar Dados de Entrada para o Modelo ---
        input_features = {}
        input_features['temperatura_media'] = daily_forecast_data.get('temperatura_media')
        input_features['umidade_media'] = daily_forecast_data.get('umidade_media')
        input_features['precipitacao_total'] = daily_forecast_data.get('precipitacao_total')
        # Para predição, usamos o nível ATUAL onde o modelo espera o nível histórico/atual
        input_features['nivel_rio_historico_ou_atual'] = current_river_level

        # Verificar se todas as features esperadas estão presentes
        if not all(feat in input_features for feat in config.ML_FEATURES):
             missing = [f for f in config.ML_FEATURES if f not in input_features]
             logging.error(f"Features faltando para predição ML: {missing}")
             return "Erro Features ML"

        # Verificar valores nulos e imputar se necessário (ex: com 0 ou média)
        for feat in config.ML_FEATURES:
            if pd.isnull(input_features[feat]):
                logging.warning(f"Valor nulo encontrado na feature '{feat}' para predição ML. Usando 0.")
                input_features[feat] = 0.0 # Imputação simples com 0

        # Criar DataFrame na ordem correta das features
        input_df = pd.DataFrame([input_features], columns=config.ML_FEATURES)

        # --- Fazer a Predição ---
        prediction = ml_model.predict(input_df)
        probability = ml_model.predict_proba(input_df)

        risk_pred = "Sim" if prediction[0] == 1 else "Não"
        risk_prob = probability[0][1] # Probabilidade de ser classe 1 ('Sim')

        logging.info(f"Predição ML (dados fictícios): {risk_pred} (Probabilidade={risk_prob:.2f})")

        return f"{risk_pred} ({risk_prob:.0%})"

    except KeyError as e:
        logging.error(f"Erro de chave ao preparar dados para predição ML: {e}. Verifique nomes das features.")
        return "Erro Features ML"
    except Exception as e:
        logging.error(f"Erro inesperado durante a predição ML: {e}")
        return "Erro Predição ML"