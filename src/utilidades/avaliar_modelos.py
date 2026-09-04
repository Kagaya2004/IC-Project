import os
import glob
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.modelos.modelo_DCRNN import DCRNNModel
from src.modelos.modelo_LSTM import LSTM_TrafficModel
from src.preprocessamento.dataLoader_Traffic import preparar_loaders

def buscar_ultimo_log(log_dir, prefixo):
    padrao = os.path.join(log_dir, f"{prefixo}*.csv")
    arquivos = glob.glob(padrao)
    return max(arquivos, key=os.path.getctime) if arquivos else None

def gerar_curva_aprendizado(log_dir, timestamp):
    print("Gerando curva de aprendizado comparativa (STGCN vs DCRNN vs LSTM)...")
    file_st = buscar_ultimo_log(log_dir, 'history_stgcn')
    file_dc = buscar_ultimo_log(log_dir, 'history_dcrnn')
    file_ls = buscar_ultimo_log(log_dir, 'history_lstm')
    
    plt.figure(figsize=(10, 6))
    
    if file_st:
        df_st = pd.read_csv(file_st)
        plt.plot(df_st['epoch'], df_st['val_mae'], color='blue', linewidth=2, label='STGCN')
    if file_dc:
        df_dc = pd.read_csv(file_dc)
        plt.plot(df_dc['epoch'], df_dc['val_mae'], color='red', linewidth=2, label='DCRNN')
    if file_ls:
        df_ls = pd.read_csv(file_ls)
        plt.plot(df_ls['epoch'], df_ls['val_mae'], color='green', linewidth=2, label='LSTM')

    plt.title('Convergência da Função de Perda durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Loss — Masked MAE (dados normalizados)')
    plt.legend(frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(log_dir, f"curva_aprendizado_{timestamp}.png"), dpi=300)
    plt.close()

def avaliar_comparativo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(log_dir, f"comparativo_triplo_{timestamp}.txt")

    # 1. Carregamento dos dados de teste sequenciais (sem leakage)
    print("Preparando DataLoaders de Teste...")
    _, test_loader, _, _ = preparar_loaders(batch_size=1, window_size=12, horizon=12)

    # 2. Carrega parâmetros de normalização salvos no treino
    scaler_path = os.path.join(data_dir, 'scaler_params.pt')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Erro: {scaler_path} não encontrado. Execute o treino primeiro!")
    
    scaler = torch.load(scaler_path, weights_only=True)
    mean_val = scaler['mean']
    std_val = scaler['std']

    # 3. Carregamento dos Modelos e Estruturas de Grafo
    print("Carregando modelos treinados...")
    
    # STGCN
    stgcn = STGCN_TrafficModel(num_features=5).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_model.pth'), weights_only=True))
    stgcn.eval()
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    # DCRNN
    adj_dc = torch.load(os.path.join(data_dir, 'adj_mx_dcrnn.pt'), weights_only=True).to(device)
    dcrnn = DCRNNModel(adj_matrices=adj_dc, in_channels=5).to(device)
    dcrnn.load_state_dict(torch.load(os.path.join(model_dir, 'dcrnn_model.pth'), weights_only=True))
    dcrnn.eval()

    # LSTM
    lstm = LSTM_TrafficModel(num_features=5).to(device)
    lstm.load_state_dict(torch.load(os.path.join(model_dir, 'lstm_model.pth'), weights_only=True))
    lstm.eval()

    # 4. Loop de Inferência e Benchmarking de Latência
    y_true, y_st, y_dc, y_ls = [], [], [], []
    lat_st, lat_dc, lat_ls = [], [], []

    print(f"Avaliando modelos em {device}...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= 500:
                break 
            x, y = x.to(device), y.to(device)
            
            # Inferência STGCN
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_st.append((time.time() - t0) * 1000)
            
            # Inferência DCRNN
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            out_dc = dcrnn(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_dc.append((time.time() - t0) * 1000)

            # Inferência LSTM (recebe [B, T, N, F] e retorna [B, Horizon, N])
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            out_ls = lstm(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_ls.append((time.time() - t0) * 1000)
            
            y_true.append(y.cpu().numpy())
            y_st.append(out_st.cpu().numpy())
            y_dc.append(out_dc.cpu().numpy())
            y_ls.append(out_ls.cpu().numpy())

    # 5. Processamento das Métricas em km/h
    y_true_np = np.array(y_true).squeeze() 
    y_st_np = np.array(y_st).squeeze()
    y_dc_np = np.array(y_dc).squeeze()
    y_ls_np = np.array(y_ls).squeeze()

    # Desnormalização exata baseada nas estatísticas do treino
    true_kmh = (y_true_np * std_val) + mean_val
    st_kmh = (y_st_np * std_val) + mean_val
    dc_kmh = (y_dc_np * std_val) + mean_val
    ls_kmh = (y_ls_np * std_val) + mean_val

    def get_metrics(pred, target):
        mask = target > 0
        mae = np.abs(pred[mask] - target[mask]).mean()
        rmse = np.sqrt(((pred[mask] - target[mask])**2).mean())
        mape = np.mean(np.abs((target[mask] - pred[mask]) / target[mask])) * 100
        return mae, rmse, mape

    metrics = {
        'STGCN': (get_metrics(st_kmh[:, 0, :], true_kmh[:, 0, :]), get_metrics(st_kmh, true_kmh), np.mean(lat_st)),
        'DCRNN': (get_metrics(dc_kmh[:, 0, :], true_kmh[:, 0, :]), get_metrics(dc_kmh, true_kmh), np.mean(lat_dc)),
        'LSTM':  (get_metrics(ls_kmh[:, 0, :], true_kmh[:, 0, :]), get_metrics(ls_kmh, true_kmh), np.mean(lat_ls))
    }

    # 6. Relatório Científico
    relatorio = (
        f"{'='*125}\n RELATÓRIO COMPARATIVO: STGCN vs DCRNN vs LSTM (60 MIN)\n{'='*125}\n"
        f"Modelo      | MAE T+1 | RMSE T+1 | MAPE T+1 | MAE (1h) | RMSE (1h) | MAPE (1h) | Latência (ms)\n"
        f"------------|---------|----------|----------|----------|-----------|-----------|---------------\n"
    )
    for mod, (t1, avg, lat) in metrics.items():
        relatorio += f"{mod:<12}| {t1[0]:.4f}  | {t1[1]:.4f}  | {t1[2]:.2f}%   | {avg[0]:.4f}  | {avg[1]:.4f}   | {avg[2]:.2f}%   | {lat:.2f}\n"
    relatorio += f"{'='*125}\n"
    
    print(relatorio)
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(relatorio)

   # 7. Gráfico Temporal (Sensor 0)
    plt.figure(figsize=(15, 6))
    
    x_axis = np.arange(150)
    plt.plot(x_axis, true_kmh[:150, 0, 0], label='Real (Ground Truth)', color='black', linewidth=2)
    plt.plot(x_axis, st_kmh[:150, 0, 0], label='STGCN', color='blue', linestyle='--')
    plt.plot(x_axis, dc_kmh[:150, 0, 0], label='DCRNN', color='red', linestyle=':')
    plt.plot(x_axis, ls_kmh[:150, 0, 0], label='LSTM', color='green', linestyle='-.')
    
    plt.title('Comparativo Temporal: Velocidade Real vs. Predita (Sensor 0)', fontsize=13, pad=10)
    plt.xlabel('Passo Temporal (intervalos de 5 minutos)', fontsize=11)
    plt.ylabel('Velocidade (km/h)', fontsize=11)
    
    # Cola o valor 0 exatamente no eixo Y e fixa o limite superior da janela
    plt.xlim(left=0, right=149)
    
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"grafico_comparativo_temporal_{timestamp}.png"), dpi=300)
    plt.close()
    
    gerar_curva_aprendizado(log_dir, timestamp)

if __name__ == "__main__":
    avaliar_comparativo()