import os
import glob
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Importações das Arquiteturas Refatoradas (Gui)
from src.modelos.modelo_STGCN_Gui import STGCN_TrafficModel
from src.modelos.modelo_DCRNN_Gui import DCRNNModel
from src.modelos.modelo_LSTM_Gui import LSTM_TrafficModel
from src.preprocessamento.dataLoader_Traffic_Gui import preparar_loaders

def buscar_ultimo_log(log_dir, prefixo):
    padrao = os.path.join(log_dir, f"{prefixo}*.csv")
    arquivos = glob.glob(padrao)
    return max(arquivos, key=os.path.getctime) if arquivos else None

def gerar_curva_aprendizado(log_dir, timestamp):
    print("📈 Gerando curva de aprendizado comparativa (STGCN vs DCRNN vs LSTM)...")
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
    plt.xlabel('Época'); plt.ylabel('Loss — Masked MAE (Normalizado)')
    plt.legend(frameon=False); plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(log_dir, f"curva_aprendizado_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def avaliar_comparativo_final():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(log_dir, f"relatorio_unificado_{timestamp}.txt")

    # 1. Carregamento Seguro (Data Leakage Free)
    print("🔄 Preparando DataLoaders e Parâmetros Z-Score...")
    # Retorna as variáveis isoladas pela nova função Gui (batch_size=32 para velocidade)
    _, test_loader, mean_val, std_val = preparar_loaders(batch_size=32, window_size=12, horizon=12)

    # 2. Carregamento dos Modelos (Interface Gui)
    print(f"🤖 Carregando os modelos na {device}...")
    
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

    y_true, y_st, y_dc, y_ls = [], [], [], []
    lat_st, lat_dc, lat_ls = [], [], []

    print("🚀 Executando passo de inferência unificada...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i > 500: break # Limite computacional para memória
            x, y = x.to(device), y.to(device)
            
            # STGCN
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_st.append((time.time() - t0) * 1000)
            
            # DCRNN
            t0 = time.time()
            out_dc = dcrnn(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_dc.append((time.time() - t0) * 1000)

            # LSTM (Invocação Limpa Graças à refatoração do Gui)
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            out_ls = lstm(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_ls.append((time.time() - t0) * 1000)
            
            y_true.append(y.cpu().numpy())
            y_st.append(out_st.cpu().numpy())
            y_dc.append(out_dc.cpu().numpy())
            y_ls.append(out_ls.cpu().numpy())

    # =================================================================================
    # PROCESSAMENTO MATRICIAL GLOBAL
    # =================================================================================
    print("🧮 Consolidando métricas e processando Desnormalização...")
    y_true_np = np.concatenate(y_true, axis=0) 
    st_np = np.concatenate(y_st, axis=0)
    dc_np = np.concatenate(y_dc, axis=0)
    ls_np = np.concatenate(y_ls, axis=0)

    # Desnormalização exata
    true_kmh = (y_true_np * std_val) + mean_val
    st_kmh = (st_np * std_val) + mean_val
    dc_kmh = (dc_np * std_val) + mean_val
    ls_kmh = (ls_np * std_val) + mean_val

    def calc_metrics(pred, target):
        mask = target > 0
        mae = np.abs(pred[mask] - target[mask]).mean()
        rmse = np.sqrt(((pred[mask] - target[mask])**2).mean())
        mape = np.mean(np.abs((target[mask] - pred[mask]) / target[mask])) * 100
        return mae, rmse, mape

    # ================= 1. RELATÓRIO GERAL (Tabela 3) =================
    metrics = {
        'ST-GCN': (calc_metrics(st_kmh[:, 0, :], true_kmh[:, 0, :]), calc_metrics(st_kmh, true_kmh), np.mean(lat_st)),
        'DCRNN':  (calc_metrics(dc_kmh[:, 0, :], true_kmh[:, 0, :]), calc_metrics(dc_kmh, true_kmh), np.mean(lat_dc)),
        'LSTM':   (calc_metrics(ls_kmh[:, 0, :], true_kmh[:, 0, :]), calc_metrics(ls_kmh, true_kmh), np.mean(lat_ls))
    }

    relatorio = (f"{'='*125}\nRELATÓRIO COMPARATIVO UNIFICADO: STGCN vs DCRNN vs LSTM (60 MIN)\n{'='*125}\n"
                 f"Modelo     | MAE T+1 | RMSE T+1 | MAPE T+1 | MAE (1h) | RMSE (1h) | MAPE (1h) | Latência (ms)\n"
                 f"-----------|---------|----------|----------|----------|-----------|-----------|---------------\n")
    for mod, (t1, avg, lat) in metrics.items():
        relatorio += f"{mod:<11}| {t1[0]:.4f}  | {t1[1]:.4f}  | {t1[2]:.2f}%  | {avg[0]:.4f}  | {avg[1]:.4f}   | {avg[2]:.2f}%   | {lat:.2f}\n"
    
    print(relatorio)
    with open(report_path, "w", encoding='utf-8') as f: f.write(relatorio)

    # ================= 2. ANÁLISE DE DEGRADAÇÃO =================
    minutos = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    mae_t_st, mae_t_dc, mae_t_ls = [], [], []

    for t in range(12):
        mask_t = true_kmh[:, t, :] > 0
        mae_t_st.append(np.abs(st_kmh[:, t, :][mask_t] - true_kmh[:, t, :][mask_t]).mean())
        mae_t_dc.append(np.abs(dc_kmh[:, t, :][mask_t] - true_kmh[:, t, :][mask_t]).mean())
        mae_t_ls.append(np.abs(ls_kmh[:, t, :][mask_t] - true_kmh[:, t, :][mask_t]).mean())

    plt.figure(figsize=(10, 6))
    plt.plot(minutos, mae_t_dc, marker='s', label='DCRNN', color='#d62728', linewidth=2)
    plt.plot(minutos, mae_t_st, marker='o', label='ST-GCN', color='#1f77b4', linewidth=2)
    plt.plot(minutos, mae_t_ls, marker='^', label='LSTM', color='#2ca02c', linewidth=2)
    plt.title('Evolução do MAE (km/h) conforme o Horizonte de Previsão')
    plt.xlabel('Minutos à frente'); plt.ylabel('MAE (km/h)'); plt.xticks(minutos)
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    plt.savefig(os.path.join(log_dir, f"curva_degradacao_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ================= 3. ANÁLISE ESPACIAL (Boxplot) =================
    mae_por_sensor_st, mae_por_sensor_dc, mae_por_sensor_ls = [], [], []
    
    for i in range(207):
        mask_i = true_kmh[:, :, i] > 0
        if mask_i.sum() == 0:
            mae_por_sensor_st.append(np.nan); mae_por_sensor_dc.append(np.nan); mae_por_sensor_ls.append(np.nan)
        else:
            mae_por_sensor_st.append(np.abs(st_kmh[:, :, i][mask_i] - true_kmh[:, :, i][mask_i]).mean())
            mae_por_sensor_dc.append(np.abs(dc_kmh[:, :, i][mask_i] - true_kmh[:, :, i][mask_i]).mean())
            mae_por_sensor_ls.append(np.abs(ls_kmh[:, :, i][mask_i] - true_kmh[:, :, i][mask_i]).mean())

    mae_st, mae_dc, mae_ls = np.array(mae_por_sensor_st), np.array(mae_por_sensor_dc), np.array(mae_por_sensor_ls)

    plt.figure(figsize=(10, 6))
    data_to_plot = [mae_st[~np.isnan(mae_st)], mae_dc[~np.isnan(mae_dc)], mae_ls[~np.isnan(mae_ls)]]
    box = plt.boxplot(data_to_plot, tick_labels=['STGCN', 'DCRNN', 'LSTM'], patch_artist=True)
    for patch in box['boxes']: patch.set(facecolor='#d3d3d3', edgecolor='black')
    for median in box['medians']: median.set(color='red', linewidth=2)
    plt.title('Distribuição do MAE Médio entre os 207 Sensores (Figura 4)')
    plt.ylabel('MAE Médio (km/h)'); plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.savefig(os.path.join(log_dir, f"boxplot_espacial_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ================= 4. MATRIZ MULTISSENSOR (Qualitativa 2x2) =================
    idx_facil = np.nanargmin(mae_st + mae_dc + mae_ls)
    idx_dificil = np.nanargmax(mae_ls - np.minimum(mae_st, mae_dc))
    erro_geral = (mae_st + mae_dc + mae_ls) / 3
    mediana = np.nanmedian(erro_geral)
    medianos = np.argsort(np.abs(erro_geral - mediana))[:2]
    
    sensores = [idx_facil, medianos[0], medianos[1], idx_dificil]
    titulos = [f'Baixa Complexidade (S_{idx_facil})', f'Padrão Mediano 1 (S_{medianos[0]})', 
               f'Padrão Mediano 2 (S_{medianos[1]})', f'Alta Complex / Vantagem GNN (S_{idx_dificil})']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Estabilidade Preditiva em Múltiplos Nós (Figura 2 Expandida)', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i, idx in enumerate(sensores):
        ax = axes[i]
        ax.plot(true_kmh[:150, 0, idx], label='Real (Ground Truth)', color='black', linewidth=2)
        ax.plot(st_kmh[:150, 0, idx], label='STGCN', color='blue', linestyle='--')
        ax.plot(dc_kmh[:150, 0, idx], label='DCRNN', color='red', linestyle=':')
        ax.plot(ls_kmh[:150, 0, idx], label='LSTM', color='green', linestyle='-.')
        ax.set_title(titulos[i]); ax.set_ylabel('Velocidade (km/h)'); ax.set_xlabel('Passos de Tempo (5 min)')
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"multissensor_2x2_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Gera a curva de aprendizado (treino vs teste) no final
    gerar_curva_aprendizado(log_dir, timestamp)

    print(f"\n✨ Execução Finalizada! Todos os artefatos foram salvos em: {log_dir}")

if __name__ == "__main__":
    avaliar_comparativo_final()