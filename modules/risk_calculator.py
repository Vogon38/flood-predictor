import numpy as np
import logging

from . import config

def calculate_historical_limits(historical_data):
    """
    Calcula limites (média, desvio padrão, percentis) a partir dos dados históricos.
    """
    if historical_data is None or historical_data.empty:
        logging.warning("Dados históricos ausentes ou vazios. Não é possível calcular limites.")
        return None

    try:
        limits = {
            'river_level_mean': historical_data['Metragem'].mean(),
            'river_level_std': historical_data['Metragem'].std(),
            'precipitation_percentile': historical_data['Volume mm'].quantile(config.PRECIPITATION_PERCENTILE_THRESHOLD) if 'Volume mm' in historical_data.columns else None,
            'days_of_rain_percentile': historical_data['Dias de chuva'].quantile(config.PRECIPITATION_PERCENTILE_THRESHOLD) if 'Dias de chuva' in historical_data.columns else None
        }
        logging.info("Limites históricos calculados:")
        logging.info(f"  Nível Rio: Média={limits['river_level_mean']:.2f}m, DP={limits['river_level_std']:.2f}m")
        if limits['precipitation_percentile'] is not None:
            logging.info(f"  Precipitação ({config.PRECIPITATION_PERCENTILE_THRESHOLD*100:.0f} Percentil): {limits['precipitation_percentile']:.2f}mm")
        if limits['days_of_rain_percentile'] is not None:
             logging.info(f"  Dias Chuva ({config.PRECIPITATION_PERCENTILE_THRESHOLD*100:.0f} Percentil): {limits['days_of_rain_percentile']:.2f} dias")

        if limits['river_level_std'] is None or limits['river_level_std'] <= 0:
            logging.warning("Desvio padrão do nível histórico do rio é zero ou inválido. O cálculo de risco baseado em DP pode não funcionar.")

        return limits
    except KeyError as e:
        logging.error(f"Erro ao calcular limites históricos: coluna '{e}' não encontrada nos dados.")
        return None
    except Exception as e:
        logging.error(f"Erro inesperado ao calcular limites históricos: {e}")
        return None


def calculate_threshold_risk(daily_forecast_data, current_river_level, historical_limits, ndwi_value=None):
    """
    Calcula o risco de enchente (1-5) e o nível estimado do rio para um dia específico,
    baseado na previsão do tempo, nível atual, limites históricos e NDWI.
    """
    if historical_limits is None:
        logging.warning("Limites históricos não disponíveis para calcular o risco.")
        return {'risk': 0, 'estimated_river_level': current_river_level, 'risk_category': 'Indeterminado'}

    try:
        chuva_prevista = daily_forecast_data.get('precipitacao_total', 0.0)
        umidade_media = daily_forecast_data.get('umidade_media', 70.0) # Usar um valor default se ausente?

        # --- Lógica de Aumento Estimado ---
        # Simplificação: Aumento proporcional à chuva prevista.
        # Pode ser MUITO mais complexo (tempo de concentração, infiltração, etc.)
        # Usar o limite de precipitação histórica como referência pode ser uma ideia.
        precip_limit = historical_limits.get('precipitation_percentile', 100) # Usar 100mm se não houver limite
        if precip_limit is None or precip_limit <= 0: precip_limit = 100 # Evitar divisão por zero

        # Fator de impacto da chuva (0 a 1+) - Quão significativa é a chuva prevista vs histórica?
        impacto_chuva_fator = chuva_prevista / precip_limit

        # Aumento estimado (exemplo MUITO simplificado):
        # Mapeia o fator de impacto para um aumento em metros. Precisa de calibração!
        # Ex: Se chover o equivalente ao percentil 80 histórico, aumenta X metros.
        # Esta relação (fator -> metros) é a parte mais difícil e específica do local.
        # VAMOS USAR UMA REGRA SIMPLES POR ENQUANTO (precisa melhorar):
        # A cada 10mm de chuva, aumenta 0.1m (apenas um chute inicial!)
        aumento_por_chuva = (chuva_prevista / 10.0) * 0.1

        # A umidade pode influenciar (solo saturado absorve menos)
        # Exemplo simples: Aumenta um pouco mais se umidade > 85%
        fator_umidade = 1.0
        if umidade_media > 85:
             fator_umidade = 1.1 # Aumenta 10% o impacto da chuva

        aumento_estimado = aumento_por_chuva * fator_umidade
        estimated_river_level = current_river_level + aumento_estimado

        # --- Ajuste por NDWI ---
        if config.USE_NDWI and ndwi_value is not None:
            if ndwi_value > config.NDWI_IMPACT_THRESHOLD:
                estimated_river_level += config.NDWI_LEVEL_ADJUSTMENT
                logging.info(f"NDWI ({ndwi_value:.3f}) > Threshold ({config.NDWI_IMPACT_THRESHOLD}). Nível ajustado em +{config.NDWI_LEVEL_ADJUSTMENT}m.")

        # --- Cálculo do Risco (Score 1-5) ---
        # Baseado em quão acima da média histórica (em termos de desvio padrão) o nível estimado está.
        mean = historical_limits.get('river_level_mean', 0)
        std = historical_limits.get('river_level_std', 1) # Usar 1 se std for inválido/zero

        if std <= 0: std = 1 # Segurança

        risk = 1 # Risco base (Normalidade)
        if estimated_river_level >= (mean + 2 * std):
            risk = 5 # Emergência
        elif estimated_river_level >= (mean + 1 * std):
            risk = 4 # Alerta
        elif estimated_river_level >= mean:
            risk = 3 # Atenção
        elif estimated_river_level >= (mean - 0.5 * std): # Abaixo da média mas não muito
            risk = 2 # Observação
        # else: risk = 1 (Normalidade)

         # Ajuste adicional de risco por NDWI
        if config.USE_NDWI and ndwi_value is not None:
             if ndwi_value > config.NDWI_IMPACT_THRESHOLD:
                 risk += config.NDWI_RISK_ADJUSTMENT
                 logging.info(f"NDWI > Threshold. Risco ajustado em +{config.NDWI_RISK_ADJUSTMENT}.")


        risk = max(1, min(risk, 5)) # Garante que o risco fique entre 1 e 5

        # Mapear score para categoria textual
        risk_categories = {
            1: "Normalidade",
            2: "Observação",
            3: "Atenção",
            4: "Alerta",
            5: "Emergência"
        }
        risk_category = risk_categories.get(risk, "Indeterminado")

        return {
            'risk_score': risk,
            'estimated_river_level': round(estimated_river_level, 2),
            'risk_category': risk_category
        }

    except Exception as e:
        logging.error(f"Erro inesperado ao calcular risco por limiar: {e}")
        # Retornar um estado seguro ou padrão
        return {'risk_score': 0, 'estimated_river_level': current_river_level, 'risk_category': 'Erro no Cálculo'}