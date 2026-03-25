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
    """Busca o log .csv mais recente com o prefixo especificado."""
    padrao = os.path.join(log_dir, f"{prefixo}_*.csv")
    arquivos = glob.glob(padrao)
    return max(arquivos, key=os.path.getctime) if arquivos else None

def gerar_curva_aprendizado(log_dir, timestamp):
    """Gera o gráfico de convergência comparando o erro de treino e validação."""
    print("📈 Buscando históricos para gerar curva de aprendizado...")
    file_st = buscar_ultimo_log(log_dir, 'history_stgcn')
    file_dc = buscar_ultimo_log(log_dir, 'history_dcrnn')

    if not file_st or not file_dc:
        print("⚠️ Aviso: Arquivos de histórico não encontrados. Verifique os nomes dos logs.")
        return

    df_st = pd.read_csv(file_st)
    df_dc = pd.read_csv(file_dc)

    plt.figure(figsize=(10, 6))
    
    # Plotagem ST-GCN (Azul)
    plt.plot(df_st['epoch'], df_st['train_mae'], label='ST-GCN (Treino)', color='blue', linestyle='--', alpha=0.5)
    plt.plot(df_st['epoch'], df_st['val_mae'], label='ST-GCN (Validação)', color='blue', linewidth=2)
    
    # Plotagem DCRNN (Vermelho)
    plt.plot(df_dc['epoch'], df_dc['train_mae'], label='DCRNN (Treino)', color='red', linestyle='--', alpha=0.5)
    plt.plot(df_dc['epoch'], df_dc['val_mae'], label='DCRNN (Validação)', color='red', linewidth=2)

    plt.title('Curvas de Aprendizado: Evolução do MAE por Época')
    plt.xlabel('Época')
    plt.ylabel('MAE (Normalizado)')
    plt.legend(frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    path_save = os.path.join(log_dir, f"curva_aprendizado_{timestamp}.png")
    plt.savefig(path_save, bbox_inches='tight', dpi=300)
    print(f"✅ Gráfico de aprendizado salvo em: {path_save}")

def avaliar_comparativo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(log_dir, f"comparativo_final_{timestamp}.txt")

    # 1. Preparação do Dataset (Horizonte 12 para ambos)
    dataset = TrafficDataset(window_size=12, horizon=12)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 2. Carregamento das Arquiteturas e Pesos
    print("🤖 Carregando modelos treinados...")
    
    # ST-GCN Multi-step
    stgcn = STGCN_TrafficModel(num_features=5).to(device)
    # Atualizado para o nome do arquivo gerado pelo novo treino
    stgcn.load_state_dict(torch.load(os.path.join(model_dir, 'stgcn_model.pth'), weights_only=True))
    stgcn.eval()
    edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)

    # DCRNN SOTA
    adj_dcrnn = torch.load(os.path.join(data_dir, 'adj_mx_dcrnn.pt'), weights_only=True).to(device)
    dcrnn = DCRNNModel(adj_matrices=adj_dcrnn, in_channels=5).to(device)
    dcrnn.load_state_dict(torch.load(os.path.join(model_dir, 'dcrnn_model.pth'), weights_only=True))
    dcrnn.eval()

    # 3. Loop de Inferência
    y_true, y_stgcn, y_dcrnn = [], [], []
    lat_stgcn, lat_dcrnn = [], []

    print(f"🚀 Iniciando inferência comparativa em {device}...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i > 500: break 
            
            x, y = x.to(device), y.to(device)
            
            # ST-GCN Latência
            t0 = time.time()
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            lat_stgcn.append((time.time() - t0) * 1000)
            
            # DCRNN Latência
            t0 = time.time()
            out_dc = dcrnn(x)
            lat_dcrnn.append((time.time() - t0) * 1000)

            y_true.append(y.cpu().numpy())
            y_stgcn.append(out_st.cpu().numpy())
            y_dcrnn.append(out_dc.cpu().numpy())

    # 4. Processamento de Dados (Horizonte 12 passos)
    y_true_np = np.array(y_true).squeeze()   # [501, 12, 207]
    y_st_np = np.array(y_stgcn).squeeze()    # [501, 12, 207]
    y_dc_np = np.array(y_dcrnn).squeeze()    # [501, 12, 207]

    # Desnormalização para km/h
    true_kmh = (y_true_np * dataset.std) + dataset.mean
    # Selecionamos o t+1 (índice 0) para o gráfico principal e métricas iniciais
    stgcn_kmh_t1 = (y_st_np[:, 0, :] * dataset.std) + dataset.mean
    dcrnn_kmh_t1 = (y_dc_np[:, 0, :] * dataset.std) + dataset.mean

    def calc_metrics(pred, target):
        mask = target > 0
        mae = np.abs(pred[mask] - target[mask]).mean()
        rmse = np.sqrt(((pred[mask] - target[mask])**2).mean())
        return mae, rmse

    # Métricas para o 1º passo (5 min à frente)
    mae_st, rmse_st = calc_metrics(stgcn_kmh_t1, true_kmh[:, 0, :])
    mae_dc, rmse_dc = calc_metrics(dcrnn_kmh_t1, true_kmh[:, 0, :])

    # 5. Relatório Textual
    relatorio = (
        f"{'='*50}\n📊 RESULTADO DA COMPARAÇÃO (T+1: 5 MIN)\n{'='*50}\n"
        f"Métrica      | ST-GCN Multi | DCRNN\n---          | ---          | ---\n"
        f"MAE (km/h)   | {mae_st:.4f}       | {mae_dc:.4f}\n"
        f"RMSE (km/h)  | {rmse_st:.4f}       | {rmse_dc:.4f}\n"
        f"Latência (ms)| {np.mean(lat_stgcn):.2f}         | {np.mean(lat_dcrnn):.2f}\n{'='*50}\n"
    )
    print(relatorio)
    with open(report_path, "w", encoding='utf-8') as f: f.write(relatorio)

    # 6. Visualização Sensor 0
    plt.figure(figsize=(15, 7))
    plt.plot(true_kmh[:150, 0, 0], label='Real (Ground Truth)', color='black', linewidth=2)
    plt.plot(stgcn_kmh_t1[:150, 0], label='Predição ST-GCN (Multi)', color='blue', linestyle='--')
    plt.plot(dcrnn_kmh_t1[:150, 0], label='Predição DCRNN', color='red', linestyle=':')
    plt.title('Comparação entre velocidades reais e preditas dos modelos STGCN e DCRNN - Sensor 0')
    plt.ylabel('Velocidade (km/h)')
    plt.xlabel('Passos de Tempo (5 min)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, f"grafico_comparativo_{timestamp}.png"))
    
    # 7. Curva de aprendizado
    gerar_curva_aprendizado(log_dir, timestamp)

if __name__ == "__main__":
    avaliar_comparativo()