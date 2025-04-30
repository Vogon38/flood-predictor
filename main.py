import logging
import pandas as pd
import os
from datetime import timedelta, date

from modules import config, data_collector, data_loader, risk_calculator, plotter, ml_predictor

def run_prediction_routine():
    """
    Executa a rotina de previsão: Limiares + Predição ML (com dados fictícios).
    NDWI/GEE está desabilitado.
    """
    logging.info("--- Iniciando Rotina de Previsão de Enchente (Limiares + ML Fictício) ---")
    if config.USE_NDWI:
         logging.warning("USE_NDWI está True em config.py, mas esta rotina assume que está desabilitado.")

    # --- Carregar Modelo ML (tentativa inicial) ---
    model_loaded = ml_predictor.ml_model is not None

    # --- 1. Carregar Dados Históricos e Calcular Limites ---
    try:
        historical_data = data_loader.load_historical_flood_data()
    except FileNotFoundError:
        logging.error(f"Arquivo histórico não encontrado em {config.HISTORICAL_FLOODS_CSV}")
        logging.error("Execute generate_fake_data.py primeiro.")
        return

    if historical_data is None or historical_data.empty:
        logging.warning("Não foi possível carregar dados históricos de 'Metragem'. Cálculo de risco por limiar será limitado.")
        historical_limits = None
    else:
        historical_limits = risk_calculator.calculate_historical_limits(historical_data)

    # --- 2. Coletar Dados Atuais ---
    logging.info("Coletando dados atuais...")
    current_river_level = data_collector.get_current_river_level()
    if current_river_level is None:
        logging.error("Falha crítica: Não foi possível obter o nível atual do rio. Usando 0 como fallback.")
        current_river_level = 0.0

    # --- NDWI Desabilitado ---
    ndwi_value = None
    logging.info("Coleta de NDWI desabilitada na configuração.")

    # --- 3. Obter Previsão do Tempo ---
    logging.info("Obtendo previsão do tempo...")
    forecast_data = data_collector.get_weather_forecast(
        config.LATITUDE, config.LONGITUDE, config.OPENWEATHER_API_KEY
    )
    if not forecast_data:
        logging.error("Falha crítica: Não foi possível obter a previsão do tempo. Abortando.")
        return

    # --- 4. Calcular Previsões Diárias (Limiar + ML) ---
    logging.info("Calculando previsões de risco diárias...")
    prediction_results = []
    sorted_dates = sorted(forecast_data.keys())
    max_days = 5

    for i, forecast_date_str in enumerate(sorted_dates):
        if i >= max_days: break
        logging.info(f"--- Processando previsão para: {forecast_date_str} ---")
        daily_data = forecast_data[forecast_date_str]

        # a) Calcular Risco por Limiar
        threshold_risk_info = risk_calculator.calculate_threshold_risk(
            daily_forecast_data=daily_data,
            current_river_level=current_river_level,
            historical_limits=historical_limits,
            ndwi_value=ndwi_value # Será None
        )

        # b) Fazer Predição com Modelo ML
        ml_prediction_result = "ML Indisponível" # Default
        if model_loaded:
            ml_prediction_result = ml_predictor.predict_ml_flood(
                daily_forecast_data=daily_data,
                current_river_level=current_river_level
            )
        else:
             logging.warning(f"Modelo ML não carregado, pulando predição ML para {forecast_date_str}.")


        # Montar resultado do dia
        result_day = {
            'data': forecast_date_str,
            'temperatura_media_prevista': daily_data.get('temperatura_media'),
            'umidade_media_prevista': daily_data.get('umidade_media'),
            'precipitacao_total_prevista': daily_data.get('precipitacao_total'),
            'descricao_tempo': daily_data.get('descricao'),
            'nivel_rio_atual_base': current_river_level,
            'ndwi_medio_utilizado': None, # NDWI desabilitado
            # Resultados do Cálculo por Limiar
            'risco_limiar_score': threshold_risk_info.get('risk_score'),
            'risco_limiar_categoria': threshold_risk_info.get('risk_category'),
            'nivel_rio_estimado_limiar': threshold_risk_info.get('estimated_river_level'),
             # Resultado da Predição ML
            'risco_ml_predicao': ml_prediction_result
        }
        prediction_results.append(result_day)

        logging.info(f"  -> Risco Limiar: {result_day['risco_limiar_categoria']} (Score {result_day['risco_limiar_score']}) | Nível Estimado: {result_day['nivel_rio_estimado_limiar']}m")
        logging.info(f"  -> Risco ML (Fictício): {result_day['risco_ml_predicao']}")


    # --- 5. Salvar Resultados ---
    if prediction_results:
        logging.info(f"Salvando resultados da previsão em {config.FORECAST_OUTPUT_CSV}")
        try:
            df_results = pd.DataFrame(prediction_results)
            # Ordem das colunas para o CSV final (adiciona coluna ML)
            cols_order = [
                'data', 'nivel_rio_atual_base', 'nivel_rio_estimado_limiar',
                'risco_limiar_score', 'risco_limiar_categoria', 'risco_ml_predicao',
                'precipitacao_total_prevista', 'temperatura_media_prevista',
                'umidade_media_prevista','ndwi_medio_utilizado', 'descricao_tempo'
            ]
            df_results = df_results[[col for col in cols_order if col in df_results.columns]]
            df_results.to_csv(config.FORECAST_OUTPUT_CSV, index=False, float_format='%.2f')
            logging.info("Resultados salvos com sucesso.")
        except Exception as e:
            logging.error(f"Erro ao salvar o arquivo CSV de resultados: {e}")

    # --- 6. Gerar Gráfico ---
    if prediction_results and historical_data is not None and current_river_level is not None:
         try:
             logging.info("Gerando gráfico de comparação de risco...")
             plotter.plot_risk_comparison(
                 current_river_level=current_river_level,
                 forecast_data=prediction_results,
                 historical_data=historical_data
             )
         except Exception as e:
              logging.error(f"Erro ao chamar a função de plotagem: {e}")

    logging.info("--- Rotina de Previsão Concluída ---")

# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    # NDWI está desabilitado na config, não precisa inicializar GEE
    run_prediction_routine()