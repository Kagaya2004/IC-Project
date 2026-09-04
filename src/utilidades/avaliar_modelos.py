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
    padrao = os.path.join(log_dir, f"{prefixo}*.csv")
    arquivos = glob.glob(padrao)
    return max(arquivos, key=os.path.getctime) if arquivos else None

def avaliar_comparativo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(log_dir, f"comparativo_triplo_final_{timestamp}.txt")

    # 1. Dataset de Teste
    dataset = TrafficDataset(window_size=12, horizon=12)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Carregando modelos treinados para inferência simultânea...")

    # Carregamento STGCN
    stgcn = STGCN_TrafficModel(num_features=5).to(device)
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_model.pth'), weights_only=True))
    stgcn.eval()
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    # Carregamento DCRNN
    adj_dc = torch.load(os.path.join(data_dir, 'adj_mx_dcrnn.pt'), weights_only=True).to(device)
    dcrnn = DCRNNModel(adj_matrices=adj_dc, in_channels=5).to(device)
    dcrnn.load_state_dict(torch.load(os.path.join(model_dir, 'dcrnn_model.pth'), weights_only=True))
    dcrnn.eval()

    # Carregamento LSTM (Descomentado e Ativo)
    lstm = LSTM_TrafficModel(num_features=5).to(device)
    lstm.load_state_dict(torch.load(os.path.join(model_dir, 'lstm_model.pth'), weights_only=True))
    lstm.eval()

    y_true, y_st, y_dc, y_ls = [], [], [], []
    lat_st, lat_dc, lat_ls = [], [], []

    print(f"Avaliando modelos em {device}...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i > 500: break # Limite de amostragem
            x, y = x.to(device), y.to(device)
            
            # Inferência STGCN
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

            # Inferência LSTM
            b, t, n, f = x.shape
            x_lstm = x.permute(0, 2, 1, 3).reshape(-1, t, f) 
            t0 = time.time()
            out_ls_raw = lstm(x_lstm) 
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_ls.append((time.time() - t0) * 1000)
            out_ls = out_ls_raw.view(b, n, 12).permute(0, 2, 1)

            y_true.append(y.cpu().numpy())
            y_st.append(out_st.cpu().numpy())
            y_dc.append(out_dc.cpu().numpy())
            y_ls.append(out_ls.cpu().numpy())

    # 2. Processamento e Desnormalização
    y_true_np = np.array(y_true).squeeze() 
    y_st_np, y_dc_np, y_ls_np = np.array(y_st).squeeze(), np.array(y_dc).squeeze(), np.array(y_ls).squeeze()

    true_kmh = (y_true_np * dataset.std) + dataset.mean
    st_kmh = (y_st_np * dataset.std) + dataset.mean
    dc_kmh = (y_dc_np * dataset.std) + dataset.mean
    ls_kmh = (y_ls_np * dataset.std) + dataset.mean

    def get_metrics(pred, target):
        mask = target > 0
        mae = np.abs(pred[mask] - target[mask]).mean()
        rmse = np.sqrt(((pred[mask] - target[mask])**2).mean())
        mape = np.mean(np.abs((target[mask] - pred[mask]) / target[mask])) * 100
        return mae, rmse, mape

    # 3. Geração de Relatório
    metrics = {
        'ST-GCN': (get_metrics(st_kmh[:, 0, :], true_kmh[:, 0, :]), get_metrics(st_kmh, true_kmh), np.mean(lat_st)),
        'DCRNN':  (get_metrics(dc_kmh[:, 0, :], true_kmh[:, 0, :]), get_metrics(dc_kmh, true_kmh), np.mean(lat_dc)),
        'LSTM':   (get_metrics(ls_kmh[:, 0, :], true_kmh[:, 0, :]), get_metrics(ls_kmh, true_kmh), np.mean(lat_ls))
    }

    relatorio = (
        f"{'='*125}\nRELATÓRIO COMPARATIVO TRIPLO: STGCN vs DCRNN vs LSTM (60 MIN)\n{'='*125}\n"
        f"Modelo     | MAE T+1 | RMSE T+1 | MAPE T+1 | MAE (1h) | RMSE (1h) | MAPE (1h) | Latência (ms)\n"
        f"-----------|---------|----------|----------|----------|-----------|-----------|---------------\n"
    )
    for mod, (t1, avg, lat) in metrics.items():
        relatorio += f"{mod:<11}| {t1[0]:.4f}  | {t1[1]:.4f}  | {t1[2]:.2f}%  | {avg[0]:.4f}  | {avg[1]:.4f}   | {avg[2]:.2f}%   | {lat:.2f}\n"
    relatorio += f"{'='*125}\n"
    
    print(relatorio)
    with open(report_path, "w", encoding='utf-8') as f: f.write(relatorio)

    # 4. Seleção Estatística Inteligente de Sensores
    print("Realizando varredura espacial para seleção estatística de nós...")
    mae_por_sensor_st, mae_por_sensor_dc, mae_por_sensor_ls = [], [], []
    num_sensores = true_kmh.shape[2]

    for i in range(num_sensores):
        mask_i = true_kmh[:, :, i] > 0
        if mask_i.sum() == 0:
            mae_por_sensor_st.append(np.inf); mae_por_sensor_dc.append(np.inf); mae_por_sensor_ls.append(np.inf)
            continue
            
        mae_por_sensor_st.append(np.abs(st_kmh[:, :, i][mask_i] - true_kmh[:, :, i][mask_i]).mean())
        mae_por_sensor_dc.append(np.abs(dc_kmh[:, :, i][mask_i] - true_kmh[:, :, i][mask_i]).mean())
        mae_por_sensor_ls.append(np.abs(ls_kmh[:, :, i][mask_i] - true_kmh[:, :, i][mask_i]).mean())

    mae_st, mae_dc, mae_ls = np.array(mae_por_sensor_st), np.array(mae_por_sensor_dc), np.array(mae_por_sensor_ls)

    idx_facil = np.nanargmin(mae_st + mae_dc + mae_ls)
    vantagem_gnn = mae_ls - np.minimum(mae_st, mae_dc)
    idx_dificil = np.nanargmax(vantagem_gnn)
    erro_medio_geral = (mae_st + mae_dc + mae_ls) / 3
    mediana_erro = np.nanmedian(erro_medio_geral)
    indices_medianos = np.argsort(np.abs(erro_medio_geral - mediana_erro))[:2]

    sensores_alvo = [idx_facil, indices_medianos[0], indices_medianos[1], idx_dificil]
    titulos_grafico = [
        f'Baixa Complexidade (Sensor {idx_facil})', 
        f'Padrão Mediano (Sensor {indices_medianos[0]})', 
        f'Padrão Mediano (Sensor {indices_medianos[1]})', 
        f'Alta Complexidade - Vantagem GNN (Sensor {idx_dificil})'
    ]

    plt.tight_layout()
    
    # Nome do arquivo de saída
    imagem_path = os.path.join(log_dir, f"grafico_multissensor_final_{timestamp}.png")
    plt.savefig(imagem_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ADICIONE ESTAS TRÊS LINHAS PARA FEEDBACK NO TERMINAL:
    print(f"✅ Varredura concluída! Sensores selecionados: {sensores_alvo}")
    print(f"📸 Gráfico 2x2 salvo com sucesso em: {imagem_path}")
    print("✨ Avaliação finalizada.")

    # 5. Plotagem da Matriz Multissensor (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Comparativo Temporal: Estabilidade Preditiva em Múltiplos Nós da Malha', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i, idx_sensor in enumerate(sensores_alvo):
        ax = axes[i]
        ax.plot(true_kmh[:150, 0, idx_sensor], label='Real (Ground Truth)', color='black', linewidth=2)
        ax.plot(st_kmh[:150, 0, idx_sensor], label='STGCN', color='blue', linestyle='--')
        ax.plot(dc_kmh[:150, 0, idx_sensor], label='DCRNN', color='red', linestyle=':')
        ax.plot(ls_kmh[:150, 0, idx_sensor], label='LSTM', color='green', linestyle='-.')
        
        ax.set_title(titulos_grafico[i])
        ax.set_ylabel('Velocidade (km/h)')
        ax.set_xlabel('Passos de Tempo (5 min)')
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"grafico_multissensor_final_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    avaliar_comparativo()