import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader_STGCN import METRLADataset
from torch.utils.data import DataLoader

def avaliar_modelo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_path = os.path.join(base_dir, 'src', 'modelos', 'stgcn_metrla.pth')
    os.makedirs(log_dir, exist_ok=True)

    # 1. Configuração do Log de Avaliação
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_log = os.path.join(log_dir, f"avaliacao_resultado_{timestamp}.txt")
    plot_path = os.path.join(log_dir, f"grafico_previsao_{timestamp}.png")

    # 2. Carregar Dados e Modelo
    dataset = METRLADataset()
    test_size = int(0.2 * len(dataset))
    _, test_ds = torch.utils.data.random_split(dataset, [len(dataset)-test_size, test_size])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr.pt'), weights_only=True).to(device)
    
    model = STGCN_TrafficModel(num_features=1).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    previsoes, reais = [], []

    print("🧐 Avaliando modelo e gerando métricas...")
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch, edge_index, edge_weight=edge_attr)
            previsoes.append(output.cpu().numpy())
            reais.append(y_batch.numpy())

    # 3. Processamento de Resultados
    previsoes = np.array(previsoes).reshape(-1, 207)
    reais = np.array(reais).reshape(-1, 207)

    # Desnormalização
    mean, std = dataset.mean, dataset.std
    previsoes_kmh = (previsoes * std) + mean
    reais_kmh = (reais * std) + mean

    # Métricas
    mae = np.mean(np.abs(reais_kmh - previsoes_kmh))
    rmse = np.sqrt(np.mean((reais_kmh - previsoes_kmh)**2))

    # 4. Guardar Log Final
    res_msg = (
        f"--- RESULTADO DA AVALIAÇÃO ({datetime.datetime.now()}) ---\n"
        f"Métrica MAE: {mae:.4f} km/h\n"
        f"Métrica RMSE: {rmse:.4f} km/h\n"
        f"Amostras de teste: {len(test_ds)}\n"
        f"------------------------------------------\n"
    )
    print(res_msg)
    with open(eval_log, "w", encoding='utf-8') as f:
        f.write(res_msg)

    # 5. Gerar e Salvar Gráfico
    sensor_id = 10 
    plt.figure(figsize=(12, 6))
    plt.plot(reais_kmh[:100, sensor_id], label='Real', color='blue', alpha=0.7)
    plt.plot(previsoes_kmh[:100, sensor_id], label='Predito', color='red', linestyle='--')
    plt.title(f'Predição de Velocidade - Sensor {sensor_id}')
    plt.ylabel('km/h')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_path)
    print(f"📈 Gráfico guardado em: {plot_path}")

if __name__ == "__main__":
    avaliar_modelo()