import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time
import pandas as pd
import glob
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.modelos.modelo_DCRNN import DCRNNModel
from src.preprocessamento.dataLoader_Traffic import TrafficDataset, DataLoader

def buscar_ultimo_log(log_dir, prefixo):
    """Busca o log .csv mais recente para compor os gráficos de aprendizado."""
    padrao = os.path.join(log_dir, f"{prefixo}*.csv")
    arquivos = glob.glob(padrao)
    return max(arquivos, key=os.path.getctime) if arquivos else None

def gerar_curva_aprendizado(log_dir, timestamp):
    print("📈 Gerando curva de aprendizado profissional...")
    file_st = buscar_ultimo_log(log_dir, 'history_stgcn')
    file_dc = buscar_ultimo_log(log_dir, 'history_dcrnn')

    if not file_st or not file_dc:
        print("⚠️ Aviso: Históricos não encontrados para o gráfico de épocas.")
        return

    df_st = pd.read_csv(file_st); df_dc = pd.read_csv(file_dc)
    plt.figure(figsize=(10, 6))
    
    # ST-GCN (Treino tracejado, Val sólido)
    plt.plot(df_st['epoch'], df_st['train_mae'], color='blue', linestyle='--', alpha=0.4, label='ST-GCN (Treino)')
    plt.plot(df_st['epoch'], df_st['val_mae'], color='blue', linewidth=2, label='ST-GCN (Validação)')
    
    # DCRNN (Treino tracejado, Val sólido)
    plt.plot(df_dc['epoch'], df_dc['train_mae'], color='red', linestyle='--', alpha=0.4, label='DCRNN (Treino)')
    plt.plot(df_dc['epoch'], df_dc['val_mae'], color='red', linewidth=2, label='DCRNN (Validação)')

    plt.title('Análise de Convergência e Estabilidade do Treinamento (ST-GCN vs. DCRNN)')
    plt.xlabel('Época')
    plt.ylabel('MAE (km/h)')
    plt.legend(frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    save_path = os.path.join(log_dir, f"curva_aprendizado_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico de convergência salvo: {save_path}")

def avaliar_comparativo():
    # Configurações de Hardware e Caminhos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(log_dir, f"comparativo_final_{timestamp}.txt")

    # 1. Dataset de Teste (Horizonte 12 para 60 min)
    dataset = TrafficDataset(window_size=12, horizon=12)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 2. Carregamento dos Modelos e Grafos
    print("🤖 Carregando modelos treinados...")
    stgcn = STGCN_TrafficModel(num_features=5).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_model.pth'), weights_only=True))
    stgcn.eval()
    
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    adj_dc = torch.load(os.path.join(data_dir, 'adj_mx_dcrnn.pt'), weights_only=True).to(device)
    dcrnn = DCRNNModel(adj_matrices=adj_dc, in_channels=5).to(device)
    dcrnn.load_state_dict(torch.load(os.path.join(model_dir, 'dcrnn_model.pth'), weights_only=True))
    dcrnn.eval()

    # 3. Inferência e Latência Otimizada
    y_true, y_st, y_dc, lat_st, lat_dc = [], [], [], [], []

    print(f"🚀 Avaliando modelos multi-step em {device}...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i > 500: break 
            x, y = x.to(device), y.to(device)
            
            # Sincronização CUDA para medição real
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_st.append((time.time() - t0) * 1000)
            
            t0 = time.time()
            out_dc = dcrnn(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_dc.append((time.time() - t0) * 1000)

            y_true.append(y.cpu().numpy()); y_st.append(out_st.cpu().numpy()); y_dc.append(out_dc.cpu().numpy())

    y_true_np = np.array(y_true).squeeze() 
    y_st_np = np.array(y_st).squeeze() 
    y_dc_np = np.array(y_dc).squeeze()

    # Desnormalização para km/h
    true_kmh = (y_true_np * dataset.std) + dataset.mean
    st_kmh = (y_st_np * dataset.std) + dataset.mean
    dc_kmh = (y_dc_np * dataset.std) + dataset.mean

    def get_metrics(pred, target):
        mask = target > 0
        mae = np.abs(pred[mask] - target[mask]).mean()
        rmse = np.sqrt(((pred[mask] - target[mask])**2).mean())
        # Cálculo do MAPE (Erro Médio Absoluto Percentual)
        mape = np.mean(np.abs((target[mask] - pred[mask]) / target[mask])) * 100
        return mae, rmse, mape

    # Coleta de métricas desmembradas por horizonte
    mae_st_t1, rmse_st_t1, mape_st_t1 = get_metrics(st_kmh[:, 0, :], true_kmh[:, 0, :])
    mae_dc_t1, rmse_dc_t1, mape_dc_t1 = get_metrics(dc_kmh[:, 0, :], true_kmh[:, 0, :])
    
    mae_st_avg, rmse_st_avg, mape_st_avg = get_metrics(st_kmh, true_kmh)
    mae_dc_avg, rmse_dc_avg, mape_dc_avg = get_metrics(dc_kmh, true_kmh)

    # 5. Relatório Científico Final Atualizado
    relatorio = (
        f"{'='*120}\n📊 RELATÓRIO DE DESEMPENHO MULTI-STEP (60 MIN)\n{'='*120}\n"
        f"Modelo     | MAE T+1 | RMSE T+1 | MAPE T+1 | MAE (1h) | RMSE (1h) | MAPE (1h) | Latência (ms)\n"
        f"-----------|---------|----------|----------|----------|-----------|-----------|---------------\n"
        f"ST-GCN     | {mae_st_t1:.4f}  | {rmse_st_t1:.4f}  | {mape_st_t1:.2f}%  | {mae_st_avg:.4f}  | {rmse_st_avg:.4f}   | {mape_st_avg:.2f}%   | {np.mean(lat_st):.2f}\n"
        f"DCRNN      | {mae_dc_t1:.4f}  | {rmse_dc_t1:.4f}  | {mape_dc_t1:.2f}%  | {mae_dc_avg:.4f}  | {rmse_dc_avg:.4f}   | {mape_dc_avg:.2f}%   | {np.mean(lat_dc):.2f}\n"
        f"{'='*120}\n"
    )
    
    print(relatorio)
    with open(report_path, "w", encoding='utf-8') as f: f.write(relatorio)

    # 6. Plotagem Gráfica de Predição
    plt.figure(figsize=(15, 6))
    plt.plot(true_kmh[:150, 0, 0], label='Real (Ground Truth)', color='black', linewidth=2)
    plt.plot(st_kmh[:150, 0, 0], label='Predição ST-GCN', color='blue', linestyle='--')
    plt.plot(dc_kmh[:150, 0, 0], label='Predição DCRNN', color='red', linestyle=':')
    plt.title('Comparativo Temporal: Velocidade Real vs. Predita - Sensor 0')
    plt.ylabel('Velocidade (km/h)'); plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(log_dir, f"grafico_predicao_{timestamp}.png"), dpi=300)
    
    gerar_curva_aprendizado(log_dir, timestamp)

if __name__ == "__main__":
    avaliar_comparativo()