import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time
import pandas as pd
import glob

# Importações dos Modelos
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.modelos.modelo_LSTM import LSTM_TrafficModel
from src.preprocessamento.dataLoader_Traffic import TrafficDataset, DataLoader

def buscar_ultimo_log(log_dir, prefixo):
    """Busca o log .csv mais recente para compor os gráficos de aprendizado."""
    padrao = os.path.join(log_dir, f"{prefixo}*.csv")
    arquivos = glob.glob(padrao)
    return max(arquivos, key=os.path.getctime) if arquivos else None

def gerar_curva_aprendizado(log_dir, timestamp):
    print("📈 Gerando curva de aprendizado comparativa (ST-GCN vs LSTM)...")
    file_st = buscar_ultimo_log(log_dir, 'history_stgcn')
    file_ls = buscar_ultimo_log(log_dir, 'history_lstm')

    if not file_st or not file_ls:
        print("⚠️ Aviso: Históricos não encontrados para o gráfico de épocas.")
        return

    df_st = pd.read_csv(file_st); df_ls = pd.read_csv(file_ls)
    plt.figure(figsize=(10, 6))
    
    # ST-GCN (Azul)
    plt.plot(df_st['epoch'], df_st['val_mae'], color='blue', linewidth=2, label='ST-GCN (Validação)')
    # LSTM (Verde)
    plt.plot(df_ls['epoch'], df_ls['val_mae'], color='green', linewidth=2, label='LSTM (Validação)')

    plt.title('Convergência: ST-GCN vs LSTM (MAE na Validação)')
    plt.xlabel('Época'); plt.ylabel('MAE (km/h)'); plt.legend(frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(log_dir, f"aprendizado_stgcn_lstm_{timestamp}.png"), dpi=300)

def avaliar_comparativo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(log_dir, f"comparativo_stgcn_lstm_{timestamp}.txt")

    # 1. Preparar Dados (Horizonte 12 para ambos)
    dataset = TrafficDataset(window_size=12, horizon=12)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 2. Carregar Modelos
    print("🤖 Carregando modelos treinados...")
    
    # ST-GCN
    stgcn = STGCN_TrafficModel(num_features=5).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_model.pth'), weights_only=True))
    stgcn.eval()
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    # LSTM (Baseline Multi-step)
    lstm = LSTM_TrafficModel(num_features=5).to(device)
    lstm.load_state_dict(torch.load(os.path.join(model_dir, 'lstm_model.pth'), weights_only=True))
    lstm.eval()

    # 3. Listas para armazenar resultados
    y_true, y_stgcn, y_lstm = [], [], []
    lat_stgcn, lat_lstm = [], []

    print(f"🚀 Iniciando inferência comparativa em {device}...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i > 500: break # ~40 horas de dados do METR-LA
            
            x, y = x.to(device), y.to(device)
            
            # --- Inferência ST-GCN ---
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_stgcn.append((time.time() - t0) * 1000)
            
            # --- Inferência LSTM (Reshape para processar todos os 207 nós) ---
            b, t, n, f = x.shape
            x_lstm = x.permute(0, 2, 1, 3).reshape(-1, t, f) # [207, 12, 5]
            
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            out_ls_raw = lstm(x_lstm) # Saída: [207, 12]
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_lstm.append((time.time() - t0) * 1000)
            
            # Reorganiza saída LSTM para [Batch, Horizon, Nodes]
            out_ls = out_ls_raw.view(b, n, 12).permute(0, 2, 1)

            # Salvar predições
            y_true.append(y.cpu().numpy())
            y_stgcn.append(out_st.cpu().numpy())
            y_lstm.append(out_ls.cpu().numpy())

    # 4. Processamento de Métricas (T+1 e Média 1h)
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

    # Métricas para 5 min (T+1) e média de 60 min (1h)
    mae_st_t1, rmse_st_t1 = calc_metrics(stgcn_kmh[:, 0, :], true_kmh[:, 0, :])
    mae_ls_t1, rmse_ls_t1 = calc_metrics(lstm_kmh[:, 0, :], true_kmh[:, 0, :])
    
    mae_st_avg, rmse_st_avg = calc_metrics(stgcn_kmh, true_kmh)
    mae_ls_avg, rmse_ls_avg = calc_metrics(lstm_kmh, true_kmh)

    # 5. Relatório Científico Completo
    relatorio = (
        f"{'='*95}\n"
        f"📊 RELATÓRIO DE IMPACTO: MODELAGEM ESPACIAL (ST-GCN) vs TEMPORAL (LSTM)\n"
        f"{'='*95}\n"
        f"Modelo     | MAE T+1 | RMSE T+1 | MAE (1h) | RMSE (1h) | Latência (ms)\n"
        f"-----------|---------|----------|----------|-----------|---------------\n"
        f"ST-GCN     | {mae_st_t1:.4f}  | {rmse_st_t1:.4f}  | {mae_st_avg:.4f}  | {rmse_st_avg:.4f}   | {np.mean(lat_stgcn):.2f}\n"
        f"LSTM       | {mae_ls_t1:.4f}  | {rmse_ls_t1:.4f}  | {mae_ls_avg:.4f}  | {rmse_ls_avg:.4f}   | {np.mean(lat_lstm):.2f}\n"
        f"{'='*95}\n"
    )
    print(relatorio)
    with open(report_path, "w", encoding='utf-8') as f: f.write(relatorio)

    # 6. Gráfico Comparativo (Sensor 0)
    plt.figure(figsize=(15, 7))
    plt.plot(true_kmh[:150, 0, 0], label='Real (Ground Truth)', color='black', linewidth=2)
    plt.plot(stgcn_kmh[:150, 0, 0], label='Predição ST-GCN', color='blue', linestyle='--')
    plt.plot(lstm_kmh[:150, 0, 0], label='Predição LSTM (Baseline)', color='green', linestyle='-.')
    
    plt.title('Impacto da Modelagem Espacial: ST-GCN vs LSTM (Sensor 0)')
    plt.ylabel('Velocidade (km/h)')
    plt.xlabel('Passos de Tempo (5 min)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, f"comparativo_temporal_stgcn_lstm_{timestamp}.png"), dpi=300)
    
    gerar_curva_aprendizado(log_dir, timestamp)

if __name__ == "__main__":
    avaliar_comparativo()