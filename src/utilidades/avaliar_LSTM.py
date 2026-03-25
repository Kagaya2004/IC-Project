import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import time
import pandas as pd

# Importações dos Modelos
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.modelos.modelo_LSTM import LSTM_TrafficModel
from src.preprocessamento.dataLoader_Traffic import TrafficDataset, DataLoader

def avaliar_comparativo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(log_dir, f"comparativo_final_{timestamp}.txt")

    # 1. Preparar Dados (Usamos o Loader Global para manter a sincronia temporal)
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

    # LSTM
    lstm = LSTM_TrafficModel(num_features=5).to(device)
    lstm.load_state_dict(torch.load(os.path.join(model_dir, 'lstm_best_baseline.pth'), weights_only=True))
    lstm.eval()

    # 3. Listas para armazenar resultados
    y_true, y_stgcn, y_lstm = [], [], []
    lat_stgcn, lat_lstm = [], []

    print(f"🚀 Iniciando inferência comparativa em {device}...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i > 500: break # Amostra para o gráfico (~40 horas)
            
            x, y = x.to(device), y.to(device)
            
            # --- Teste ST-GCN ---
            t0 = time.time()
            out_st = stgcn(x, edge_index, edge_weight=edge_attr)
            lat_stgcn.append((time.time() - t0) * 1000)
            
            # --- Teste LSTM ---
            t0 = time.time()
            # Transformamos [Batch=1, Time=12, Nodes=207, Feats=5]
            # em [Batch=207, Time=12, Feats=5]. A LSTM prevê todos os nós de uma vez!
            x_lstm = x.squeeze(0).permute(1, 0, 2) 
            out_ls = lstm(x_lstm) # Saída: [207]
            out_ls = out_ls.unsqueeze(0) # Voltamos para [1, 207] para casar com o target
            lat_lstm.append((time.time() - t0) * 1000)

            # Salvar predições
            y_true.append(y.cpu().numpy())
            y_stgcn.append(out_st.cpu().numpy())
            y_lstm.append(out_ls.cpu().numpy())

    # 4. Processamento de Métricas (Considerando apenas o passo t+1)
    y_true_np = np.array(y_true).squeeze()
    y_stgcn_np = np.array(y_stgcn).squeeze()
    y_lstm_np = np.array(y_lstm).squeeze()

    # Desnormalização
    true_kmh = (y_true_np * dataset.std) + dataset.mean
    stgcn_kmh = (y_stgcn_np * dataset.std) + dataset.mean
    lstm_kmh = (y_lstm_np * dataset.std) + dataset.mean

    def calc_metrics(pred, target):
        mask = target > 0
        mae = np.abs(pred[mask] - target[mask]).mean()
        rmse = np.sqrt(((pred[mask] - target[mask])**2).mean())
        return mae, rmse

    mae_st, rmse_st = calc_metrics(stgcn_kmh, true_kmh[:, 0, :])
    mae_ls, rmse_ls = calc_metrics(lstm_kmh, true_kmh[:, 0, :])

    # 5. Relatório Atualizado
    relatorio = (
        f"{'='*50}\n"
        f"📊 RESULTADO DA COMPARAÇÃO CIENTÍFICA\n"
        f"{'='*50}\n"
        f"Métrica      | ST-GCN  | LSTM (Baseline)\n"
        f"-------------|---------|----------------\n"
        f"MAE (km/h)   | {mae_st:.4f}  | {mae_ls:.4f}\n"
        f"RMSE (km/h)  | {rmse_st:.4f}  | {rmse_ls:.4f}\n"
        f"Latência (ms)| {np.mean(lat_stgcn):.2f}   | {np.mean(lat_lstm):.2f}\n"
        f"{'='*50}\n"
    )
    print(relatorio)
    with open(report_path, "w", encoding='utf-8') as f: f.write(relatorio)

    # 6. Gráfico Comparativo (Sensor 0)
    plt.figure(figsize=(15, 7))
    plt.plot(true_kmh[:150, 0, 0], label='Real (Ground Truth)', color='black', linewidth=2)
    plt.plot(stgcn_kmh[:150, 0], label='Predição ST-GCN', color='blue', linestyle='--')
    plt.plot(lstm_kmh[:150, 0], label='Predição LSTM (Baseline)', color='green', linestyle='-.')
    
    plt.title('Impacto da Modelagem Espacial: ST-GCN vs LSTM')
    plt.ylabel('Velocidade (km/h)')
    plt.xlabel('Passos de Tempo (5 min)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, f"grafico_comparativo_espacial_{timestamp}.png"))
    print(f"✅ Gráfico salvo em: {log_dir}")

if __name__ == "__main__":
    avaliar_comparativo()