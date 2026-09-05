import os
import glob
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Importações das Arquiteturas Refatoradas (Gui)
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.modelos.modelo_DCRNN import DCRNNModel
from src.modelos.modelo_LSTM import LSTM_TrafficModel
from src.preprocessamento.dataLoader_Traffic import preparar_loaders

def buscar_ultimo_log(log_dir, prefixo):
    """Busca o arquivo de log mais recente para plotagem."""
    padrao = os.path.join(log_dir, f"{prefixo}*.csv")
    arquivos = glob.glob(padrao)
    return max(arquivos, key=os.path.getctime) if arquivos else None

def gerar_curva_aprendizado(log_dir, timestamp):
    print("\n📈 Gerando curva de aprendizado comparativa (STGCN vs DCRNN vs LSTM)...")
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
    plt.ylabel('Loss — Masked MAE (Normalizado)')
    plt.legend(frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Salva e notifica
    plt.savefig(os.path.join(log_dir, f"curva_aprendizado_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 Gráfico salvo: curva_aprendizado_{timestamp}.png")

def avaliar_tudo_unificado():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(log_dir, f"relatorio_unificado_final_{timestamp}.txt")

    # 1. Preparação Segura de Dados
    print("🔄 Preparando DataLoaders e Parâmetros Z-Score (Sem Vazamento)...")
    _, test_loader, train_mean, train_std = preparar_loaders(batch_size=32, window_size=12, horizon=12)

    # 2. Carregamento dos Modelos Otimizados
    print(f"🤖 Carregando os três modelos treinados em {device}...")
    
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

    total_batches = len(test_loader)
    print(f"🚀 Executando inferência unificada em toda a base de teste ({total_batches} lotes)...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
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

            # LSTM 
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            out_ls = lstm(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            lat_ls.append((time.time() - t0) * 1000)
            
            # Extração dos tensores para a CPU
            y_true.append(y.cpu().numpy())
            y_st.append(out_st.cpu().numpy())
            y_dc.append(out_dc.cpu().numpy())
            y_ls.append(out_ls.cpu().numpy())
            
            # Feedback dinâmico de progresso no terminal
            if (i + 1) % 20 == 0 or (i + 1) == total_batches:
                print(f"⏳ Processando lote {i + 1}/{total_batches}...", end='\r')

    print("\n✅ Inferência concluída com sucesso.")

    # =================================================================================
    # PROCESSAMENTO MATRICIAL GLOBAL
    # =================================================================================
    print("🧮 Consolidando métricas e processando Desnormalização...")
    y_true_np = np.concatenate(y_true, axis=0) 
    st_np = np.concatenate(y_st, axis=0)
    dc_np = np.concatenate(y_dc, axis=0)
    ls_np = np.concatenate(y_ls, axis=0)

    # Desnormalização exata
    true_kmh = (y_true_np * train_std) + train_mean
    st_kmh = (st_np * train_std) + train_mean
    dc_kmh = (dc_np * train_std) + train_mean
    ls_kmh = (ls_np * train_std) + train_mean

    def calc_metrics(pred, target):
        mask = target > 0
        mae = np.abs(pred[mask] - target[mask]).mean()
        rmse = np.sqrt(((pred[mask] - target[mask])**2).mean())
        mape = np.mean(np.abs((target[mask] - pred[mask]) / target[mask])) * 100
        return mae, rmse, mape

    # ================= 1. RELATÓRIO GERAL (Tabela 3) =================
    print("📊 Gerando Relatório Analítico (Tabela 3)...")
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
    print("📉 Gerando Curva de Degradação Temporal...")
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
    plt.xlabel('Minutos à frente')
    plt.ylabel('MAE (km/h)')
    plt.xticks(minutos)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Salva e notifica
    plt.savefig(os.path.join(log_dir, f"curva_degradacao_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 Gráfico salvo: curva_degradacao_{timestamp}.png")

    # ================= 3. ANÁLISE ESPACIAL (Boxplot e Scatter) =================
    print("🗺️ Gerando Dispersão Espacial (Scatter e Boxplot com proteção estatística)...")
    mae_por_sensor_st, mae_por_sensor_dc, mae_por_sensor_ls = [], [], []
    num_sensores = true_kmh.shape[2]
    
    for i in range(num_sensores):
        mask_i = true_kmh[:, :, i] > 0
        if mask_i.sum() == 0:
            mae_por_sensor_st.append(np.nan)
            mae_por_sensor_dc.append(np.nan)
            mae_por_sensor_ls.append(np.nan)
        else:
            mae_por_sensor_st.append(np.abs(st_kmh[:, :, i][mask_i] - true_kmh[:, :, i][mask_i]).mean())
            mae_por_sensor_dc.append(np.abs(dc_kmh[:, :, i][mask_i] - true_kmh[:, :, i][mask_i]).mean())
            mae_por_sensor_ls.append(np.abs(ls_kmh[:, :, i][mask_i] - true_kmh[:, :, i][mask_i]).mean())

    mae_st, mae_dc, mae_ls = np.array(mae_por_sensor_st), np.array(mae_por_sensor_dc), np.array(mae_por_sensor_ls)

    # Boxplot (Figura 4)
    plt.figure(figsize=(10, 6))
    data_to_plot = [mae_st[~np.isnan(mae_st)], mae_dc[~np.isnan(mae_dc)], mae_ls[~np.isnan(mae_ls)]]
    box = plt.boxplot(data_to_plot, tick_labels=['STGCN', 'DCRNN', 'LSTM'], patch_artist=True)
    for patch in box['boxes']: patch.set(facecolor='#d3d3d3', edgecolor='black')
    for median in box['medians']: median.set(color='red', linewidth=2)
    plt.title('Distribuição do MAE Médio entre os 207 Sensores (Figura 4)')
    plt.ylabel('MAE Médio (km/h)')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Salva e notifica
    plt.savefig(os.path.join(log_dir, f"boxplot_espacial_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 Gráfico salvo: boxplot_espacial_{timestamp}.png")

    # Scatter Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(range(num_sensores), mae_st, color='blue', alpha=0.6, s=20, label='STGCN')
    plt.scatter(range(num_sensores), mae_dc, color='red', alpha=0.6, s=20, label='DCRNN')
    plt.scatter(range(num_sensores), mae_ls, color='green', alpha=0.6, s=20, label='LSTM')
    plt.title('Comparativo do Erro MAE Individual entre os 207 Sensores')
    plt.xlabel('Índice do Sensor')
    plt.ylabel('MAE Médio (km/h)')
    plt.xlim(0, num_sensores - 1)
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Salva e notifica
    plt.savefig(os.path.join(log_dir, f"scatter_espacial_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 Gráfico salvo: scatter_espacial_{timestamp}.png")

    # ================= 4. MATRIZ MULTISSENSOR (Qualitativa 2x2) =================
    print("🎯 Selecionando sensores inteligentemente e gerando Matriz 2x2...")
    idx_facil = np.nanargmin(mae_st + mae_dc + mae_ls)
    idx_dificil = np.nanargmax(mae_ls - np.minimum(mae_st, mae_dc))
    erro_geral = (mae_st + mae_dc + mae_ls) / 3
    mediana = np.nanmedian(erro_geral)
    medianos = np.argsort(np.abs(erro_geral - mediana))[:2]
    
    sensores = [idx_facil, medianos[0], medianos[1], idx_dificil]
    titulos = [f'Baixa Complexidade (S_{idx_facil})', f'Padrão Mediano 1 (S_{medianos[0]})', 
               f'Padrão Mediano 2 (S_{medianos[1]})', f'Alta Complexidade / Vantagem GNN (S_{idx_dificil})']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Estabilidade Preditiva em Múltiplos Nós (Figura 2 Expandida)', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    # 1. CRIAÇÃO DO EIXO X EM MINUTOS
    # Cria um vetor de 150 posições [0, 1, 2...] e multiplica por 5
    # Resultado: [0, 5, 10, 15... 745] minutos
    x_minutos = np.arange(150) * 5

    for i, idx in enumerate(sensores):
        ax = axes[i]
        
        # 2. INJEÇÃO DA VARIÁVEL 'x_minutos' COMO PRIMEIRO ARGUMENTO DO PLOT
        ax.plot(x_minutos, true_kmh[:150, 0, idx], label='Real (Ground Truth)', color='black', linewidth=2)
        ax.plot(x_minutos, st_kmh[:150, 0, idx], label='STGCN', color='blue', linestyle='--')
        ax.plot(x_minutos, dc_kmh[:150, 0, idx], label='DCRNN', color='red', linestyle=':')
        ax.plot(x_minutos, ls_kmh[:150, 0, idx], label='LSTM', color='green', linestyle='-.')
        
        ax.set_title(titulos[i])
        ax.set_ylabel('Velocidade (km/h)')
        
        # 3. ATUALIZAÇÃO DO RÓTULO DO EIXO X
        ax.set_xlabel('Tempo Decorrido (Minutos)') 
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=12)
    plt.tight_layout()
    
    # Salva e notifica
    plt.savefig(os.path.join(log_dir, f"multissensor_2x2_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 Gráfico salvo: multissensor_2x2_{timestamp}.png")

    # Gera a curva de aprendizado por último
    gerar_curva_aprendizado(log_dir, timestamp)

    print(f"\n✨ Execução Finalizada com Sucesso!")
    print(f"📁 Todos os artefatos visuais e o relatório final foram salvos no diretório: {log_dir}")

if __name__ == "__main__":
    avaliar_tudo_unificado()