import matplotlib.pyplot as plt
import logging
import os
import pandas as pd

from . import config

def plot_risk_comparison(current_river_level, forecast_data, historical_data, output_filename="flood_risk_comparison.png"):
    """
    Gera e salva um gráfico comparando níveis atuais, estimados e históricos.
    forecast_data: Lista de dicionários, cada um com 'data' e 'nivel_rio_estimado_limiar'.
    historical_data: DataFrame com a coluna 'Metragem'.
    """
    if historical_data is None or historical_data.empty:
        logging.warning("Dados históricos não disponíveis para plotagem.")
        return
    if not forecast_data:
        logging.warning("Dados de previsão não disponíveis para plotagem.")
        return

    try:
        plt.figure(figsize=(12, 6))

        # Histograma dos níveis históricos
        if 'Metragem' in historical_data.columns:
            plt.hist(historical_data['Metragem'].dropna(), bins=20, alpha=0.6, label='Histórico Níveis Rio', color='skyblue', density=True)
            mean_hist = historical_data['Metragem'].mean()
            std_hist = historical_data['Metragem'].std()

            # Linhas de referência histórica
            plt.axvline(mean_hist, color='green', linestyle='--', linewidth=1, label=f'Média Hist ({mean_hist:.2f}m)')
            if std_hist > 0:
                plt.axvline(mean_hist + std_hist, color='darkorange', linestyle=':', linewidth=1, label=f'Média+1DP ({mean_hist + std_hist:.2f}m)')
                plt.axvline(mean_hist + 2*std_hist, color='red', linestyle=':', linewidth=1, label=f'Média+2DP ({mean_hist + 2*std_hist:.2f}m)')
        else:
            logging.warning("Coluna 'Metragem' não encontrada nos dados históricos para o histograma.")

        # Linha do nível atual
        plt.axvline(current_river_level, color='blue', linestyle='-', linewidth=2, label=f'Nível Atual ({current_river_level:.2f}m)')

        # Pontos ou linha para níveis estimados futuros (usando o cálculo por limiar)
        dates = [d['data'] for d in forecast_data]
        estimated_levels = [d.get('nivel_rio_estimado_limiar') for d in forecast_data]

        # Filtrar dias sem estimativa válida
        valid_forecast = [(d, l) for d, l in zip(dates, estimated_levels) if l is not None]
        if valid_forecast:
             dates_valid = [item[0] for item in valid_forecast]
             levels_valid = [item[1] for item in valid_forecast]
             # Usar scatter para mostrar pontos individuais estimados
             plt.scatter(levels_valid, [0.01]*len(levels_valid), color='red', s=80, zorder=5, label='Nível Estimado (Limiar)')
             # Adicionar texto com a data perto dos pontos
             # for i, level in enumerate(levels_valid):
             #     plt.text(level, 0.015 + i*0.005, f"{dates_valid[i][-5:]}", fontsize=8, ha='center') # Mostra MM-DD


        plt.title('Nível do Rio: Atual vs. Estimado (Limiar) vs. Histórico')
        plt.xlabel('Nível do rio (metros)')
        plt.ylabel('Densidade (Histograma) / Marcadores (Estimativas)')
        plt.legend(loc='upper right')
        plt.grid(True, axis='x')
        plt.ylim(bottom=0) # Garante que o eixo Y comece em 0 para o histograma
        plt.tight_layout()

        # Salvar o gráfico
        output_path = os.path.join(config.DATA_DIR, output_filename)
        plt.savefig(output_path)
        logging.info(f"Gráfico de comparação de risco salvo em: {output_path}")
        plt.close() # Fecha a figura para liberar memória

    except Exception as e:
        logging.error(f"Erro ao gerar ou salvar o gráfico de risco: {e}")