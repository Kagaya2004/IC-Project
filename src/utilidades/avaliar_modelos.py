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
from src.modelos.modelo_DCRNN import DCRNNModel
from src.modelos.modelo_LSTM import LSTM_TrafficModel
from src.preprocessamento.dataLoader_Traffic import TrafficDataset, DataLoader

def buscar_ultimo_log(log_dir, prefixo):
    # Define caminho para arquivos CSV com o prefixo específico
    padrao = os.path.join(log_dir, f"{prefixo}*.csv")
    # Busca os arquivos correspondentes e retorna o mais recente
    arquivos = glob.glob(padrao)
    return max(arquivos, key=os.path.getctime) if arquivos else None

def gerar_curva_aprendizado(log_dir, timestamp):
    print("Gerando curva de aprendizado comparativa (ST-GCN vs DCRNN vs LSTM)...")
    # Busca os arquivos de log mais recentes para cada modelo
    file_st = buscar_ultimo_log(log_dir, 'history_stgcn')
    file_dc = buscar_ultimo_log(log_dir, 'history_dcrnn')
    file_ls = buscar_ultimo_log(log_dir, 'history_lstm')

    # Configurações de plotagem
    plt.figure(figsize=(10, 6))
    
    # Plotagem das curvas de validação
    if file_st:
        df_st = pd.read_csv(file_st)
        plt.plot(df_st['epoch'], df_st['val_mae'], color='blue', linewidth=2, label='ST-GCN')
    if file_dc:
        df_dc = pd.read_csv(file_dc)
        plt.plot(df_dc['epoch'], df_dc['val_mae'], color='red', linewidth=2, label='DCRNN')
    """
    if file_ls:
        df_ls = pd.read_csv(file_ls)
        plt.plot(df_ls['epoch'], df_ls['val_mae'], color='green', linewidth=2, label='LSTM')
    """
        
    plt.title('Análise de Convergência e Estabilidade do Treinamento (ST-GCN vs. DCRNN)')
    plt.xlabel('Época'); plt.ylabel('MAE (km/h)'); plt.legend(frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(log_dir, f"curva_aprendizado_{timestamp}.png"), dpi=300)

def avaliar_comparativo():
    # Define o dispositivo (GPU se disponível, caso contrário CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Raiz do projeto
    base_dir = r'C:\IC\Projeto_V2'
    # Diretório de dados tratados
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    # Diretório de logs
    log_dir = os.path.join(base_dir, 'data', 'logs')
    # Diretório  de modelos
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    # Identificação única do experimento para histórico dos logs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(log_dir, f"comparativo_triplo_{timestamp}.txt")

    # Dataset de Teste (Horizonte 12 para 60 min)
    dataset = TrafficDataset(window_size=12, horizon=12)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Carregamento dos Modelos e Grafos
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

    """
    # LSTM
    lstm = LSTM_TrafficModel(num_features=5).to(device)
    lstm.load_state_dict(torch.load(os.path.join(model_dir, 'lstm_model.pth'), weights_only=True))
    lstm.eval()
    """

    # Inferência Unificada e Latência
    y_true, y_st, y_dc, y_ls = [], [], [], []
    lat_st, lat_dc, lat_ls = [], [], []

    print(f"🚀 Avaliando modelos em {device}...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i > 500: break 
            x, y = x.to(device), y.to(device)
            
            # Inferência do STGCN
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_st.append((time.time() - t0) * 1000)
            
            # Inferência DCRNN
            t0 = time.time()
            out_dc = dcrnn(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_dc.append((time.time() - t0) * 1000)

            """
            # Inferência LSTM
            b, t, n, f = x.shape
            x_lstm = x.permute(0, 2, 1, 3).reshape(-1, t, f) 
            t0 = time.time()
            out_ls_raw = lstm(x_lstm) 
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_ls.append((time.time() - t0) * 1000)
            out_ls = out_ls_raw.view(b, n, 12).permute(0, 2, 1)
            """

            # Armazenar resultados
            y_true.append(y.cpu().numpy())
            y_st.append(out_st.cpu().numpy())
            y_dc.append(out_dc.cpu().numpy())
            # y_ls.append(out_ls.cpu().numpy())

    # Processamento de Métricas
    y_true_np = np.array(y_true).squeeze() 
    y_st_np, y_dc_np, y_ls_np = np.array(y_st).squeeze(), np.array(y_dc).squeeze(), np.array(y_ls).squeeze()

    # Desnormalização para km/h
    true_kmh = (y_true_np * dataset.std) + dataset.mean
    st_kmh = (y_st_np * dataset.std) + dataset.mean
    dc_kmh = (y_dc_np * dataset.std) + dataset.mean
    # ls_kmh = (y_ls_np * dataset.std) + dataset.mean

    def get_metrics(pred, target):
        mask = target > 0
        mae = np.abs(pred[mask] - target[mask]).mean()
        rmse = np.sqrt(((pred[mask] - target[mask])**2).mean())
        mape = np.mean(np.abs((target[mask] - pred[mask]) / target[mask])) * 100
        return mae, rmse, mape

    # Coleta de métricas (T+1 e Média 1h)
    metrics = {
        'ST-GCN': (get_metrics(st_kmh[:, 0, :], true_kmh[:, 0, :]), get_metrics(st_kmh, true_kmh), np.mean(lat_st)),
        'DCRNN':  (get_metrics(dc_kmh[:, 0, :], true_kmh[:, 0, :]), get_metrics(dc_kmh, true_kmh), np.mean(lat_dc)),
        # 'LSTM':   (get_metrics(ls_kmh[:, 0, :], true_kmh[:, 0, :]), get_metrics(ls_kmh, true_kmh), np.mean(lat_ls))
    }

    # 5. Relatório Científico Final
    relatorio = (
        f"{'='*125}\n📊 RELATÓRIO COMPARATIVO TRIPLO: ST-GCN vs DCRNN vs LSTM (60 MIN)\n{'='*125}\n"
        f"Modelo     | MAE T+1 | RMSE T+1 | MAPE T+1 | MAE (1h) | RMSE (1h) | MAPE (1h) | Latência (ms)\n"
        f"-----------|---------|----------|----------|----------|-----------|-----------|---------------\n"
    )
    for mod, (t1, avg, lat) in metrics.items():
        relatorio += f"{mod:<11}| {t1[0]:.4f}  | {t1[1]:.4f}  | {t1[2]:.2f}%  | {avg[0]:.4f}  | {avg[1]:.4f}   | {avg[2]:.2f}%   | {lat:.2f}\n"
    relatorio += f"{'='*125}\n"
    
    print(relatorio)
    with open(report_path, "w", encoding='utf-8') as f: f.write(relatorio)

    # 6. Plotagem de Predição (Sensor 0)
    plt.figure(figsize=(15, 6))
    plt.plot(true_kmh[:150, 0, 0], label='Real (Ground Truth)', color='black', linewidth=2)
    plt.plot(st_kmh[:150, 0, 0], label='ST-GCN', color='blue', linestyle='--')
    plt.plot(dc_kmh[:150, 0, 0], label='DCRNN', color='red', linestyle=':')
    # plt.plot(ls_kmh[:150, 0, 0], label='LSTM', color='green', linestyle='-.')
    plt.title('Comparativo Temporal: Velocidade Real vs. Predita (Sensor 0)')
    plt.ylabel('Velocidade (km/h)'); plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(log_dir, f"grafico_comparativo_temporal_{timestamp}.png"), dpi=300)
    
    gerar_curva_aprendizado(log_dir, timestamp)

if __name__ == "__main__":
    avaliar_comparativo()