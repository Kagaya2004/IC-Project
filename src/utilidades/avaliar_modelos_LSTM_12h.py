import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time
import pandas as pd
import glob

# Importa as versões de 12 horas (12 passos) dos modelos
from src.modelos.modelo_STGCN_12h import STGCN_TrafficModel_12h
from src.modelos.modelo_LSTM import LSTMModel
from src.preprocessamento.dataLoader_Traffic import TrafficDataset, DataLoader

def buscar_ultimo_log(log_dir, prefixo):
    # Busca o log .csv mais recente com o prefixo especificado no diretório de logs.
    padrao = os.path.join(log_dir, f"{prefixo}_*.csv")
    arquivos = glob.glob(padrao)
    return max(arquivos, key=os.path.getctime) if arquivos else None

def gerar_curva_aprendizado(log_dir, timestamp):
    print("📈 Buscando históricos para gerar curva de aprendizado...")
    # Busca os históricos ESPECÍFICOS do treinamento de 12 passos
    file_st = buscar_ultimo_log(log_dir, 'history_stgcn_12h')
    file_lstm = buscar_ultimo_log(log_dir, 'history_lstm_12h')

    if not file_st or not file_lstm:
        print("⚠️ Aviso: Arquivos de histórico (history_*_12h.csv) não encontrados.")
        return

    df_st = pd.read_csv(file_st)
    df_lstm = pd.read_csv(file_lstm)

    plt.figure(figsize=(10, 6))
    
    # Plotagem ST-GCN (Azul)
    plt.plot(df_st['epoch'], df_st['train_mae'], label='ST-GCN (Treino)', color='blue', linestyle='--', alpha=0.5)
    plt.plot(df_st['epoch'], df_st['val_mae'], label='ST-GCN (Validação)', color='blue', linewidth=2)
    
    # Plotagem LSTM (Verde)
    plt.plot(df_lstm['epoch'], df_lstm['train_mae'], label='LSTM (Treino)', color='green', linestyle='--', alpha=0.5)
    plt.plot(df_lstm['epoch'], df_lstm['val_mae'], label='LSTM (Validação)', color='green', linewidth=2)

    plt.title('Curvas de aprendizado (Longo Prazo): evolução do MAE Médio (1 Hora)')
    plt.xlabel('Época')
    plt.ylabel('MAE (km/h)')
    plt.legend(frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    path_save = os.path.join(log_dir, f"curva_aprendizado_12h_{timestamp}.png")
    plt.savefig(path_save, bbox_inches='tight', dpi=300)
    print(f"✅ Gráfico de aprendizado salvo em: {path_save}")

def avaliar_comparativo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(log_dir, f"comparativo_final_12h_{timestamp}.txt")

    # 1. Preparação do Dataset
    dataset = TrafficDataset(window_size=12, horizon=12)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 2. Carregamento das Arquiteturas e Pesos Treinados (Versão 12h)
    print("🤖 Carregando modelos treinados (1 Hora)...")
    
    # ST-GCN: Versão ajustada para 12 passos
    stgcn = STGCN_TrafficModel_12h(num_features=5, horizon=12).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_model_12h.pth'), weights_only=True))
    stgcn.eval()
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    # LSTM: Instanciado com horizon=12
    lstm = LSTMModel(input_dim=5, hidden_dim=64, num_layers=2, horizon=12).to(device)
    lstm.load_state_dict(torch.load(os.path.join(model_dir, 'lstm_model_12h.pth'), weights_only=True))
    lstm.eval()

    # 3. Loop de Inferência
    y_true, y_stgcn, y_lstm = [], [], []
    lat_stgcn, lat_lstm = [], []

    print(f"🚀 Iniciando inferência comparativa em {device}...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i > 500: break # Limita para não pesar a memória na geração de gráficos
            
            x, y = x.to(device), y.to(device)
            
            t0 = time.time()
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            lat_stgcn.append((time.time() - t0) * 1000)
            
            t0 = time.time()
            out_lstm = lstm(x)
            lat_lstm.append((time.time() - t0) * 1000)

            y_true.append(y.cpu().numpy())
            y_stgcn.append(out_st.cpu().numpy())
            y_lstm.append(out_lstm.cpu().numpy())

    # 4. Processamento de Dados
    # Os arrays agora são [Amostras, Horizonte(12), Sensores(207)]
    y_true_np = np.array(y_true).squeeze()
    y_stgcn_np = np.array(y_stgcn).squeeze()
    y_lstm_np = np.array(y_lstm).squeeze()

    # Desnormalização para km/h
    true_kmh = (y_true_np * dataset.std) + dataset.mean
    stgcn_kmh = (y_stgcn_np * dataset.std) + dataset.mean
    lstm_kmh = (y_lstm_np * dataset.std) + dataset.mean

    def calc_metrics(pred, target):
        mask = target > 0
        mae = np.abs(pred[mask] - target[mask]).mean()
        rmse = np.sqrt(((pred[mask] - target[mask])**2).mean())
        return mae, rmse

    # Cálculo final englobando a média de todos os 12 passos
    mae_st, rmse_st = calc_metrics(stgcn_kmh, true_kmh)
    mae_lstm, rmse_lstm = calc_metrics(lstm_kmh, true_kmh)

    # 5. Geração do Relatório
    relatorio = (
        f"{'='*60}\n📊 RESULTADO DA COMPARAÇÃO (PREVISÃO DE 1 HORA)\n{'='*60}\n"
        f"Métrica | ST-GCN | LSTM\n--- | --- | ---\n"
        f"MAE Médio (km/h) | {mae_st:.4f} | {mae_lstm:.4f}\n"
        f"RMSE Médio (km/h)| {rmse_st:.4f} | {rmse_lstm:.4f}\n"
        f"Latência (ms) | {np.mean(lat_stgcn):.2f} | {np.mean(lat_lstm):.2f}\n"
        f"{'='*60}\n* Nota: O erro representa a média de todos os 12 passos projetados.\n"
    )
    print(relatorio)
    with open(report_path, "w", encoding='utf-8') as f: f.write(relatorio)

    # 6. Visualização de Série Temporal
    plt.figure(figsize=(15, 7))
    
    # Para visualização contínua, extraímos apenas o primeiro passo do horizonte (índice 0)
    # Dimensões: [Amostras, Horizonte, Sensores] -> Pegamos [:150, 0, 0] (150 amostras, 1º passo, 1º Sensor)
    y_plot_true = true_kmh[:150, 0, 0]
    y_plot_st = stgcn_kmh[:150, 0, 0]
    y_plot_lstm = lstm_kmh[:150, 0, 0]

    plt.plot(y_plot_true, label='Real (Ground Truth)', color='black', linewidth=2)
    plt.plot(y_plot_st, label='Predição ST-GCN', color='blue', linestyle='--')
    plt.plot(y_plot_lstm, label='Predição LSTM', color='green', linestyle=':')
    
    plt.title('Comparação de Modelos - Previsão Contínua (Visualizando passo t+1) - Sensor 0')
    plt.ylabel('Velocidade (km/h)')
    plt.xlabel('Passos de Tempo (5 min)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, f"grafico_serie_12h_{timestamp}.png"))
    
    # 7. Gera curva de aprendizado
    gerar_curva_aprendizado(log_dir, timestamp)

if __name__ == "__main__":
    avaliar_comparativo()