import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader_STGCN import METRLADataset
from torch.utils.data import DataLoader

def avaliar_modelo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Caminhos
    data_dir = r'C:\IC\Projeto_V2\data\tratados'
    model_path = r'C:\IC\Projeto_V2\src\modelos\stgcn_metrla.pth'
    
    # 2. Carregar o Dataset (para pegar as médias de desnormalização)
    # É vital usar os mesmos parâmetros do treino
    dataset_teste = METRLADataset()
    # Usaremos apenas os últimos 20% para teste real
    test_size = int(0.2 * len(dataset_teste))
    _, test_ds = torch.utils.data.random_split(dataset_teste, [len(dataset_teste)-test_size, test_size])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 3. Carregar o Grafo e o Modelo
    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt')).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr.pt')).to(device)
    
    model = STGCN_TrafficModel(num_features=1).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    previsoes = []
    reais = []

    print("🧐 Iniciando avaliação em dados não vistos...")
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            
            # Predição
            output = model(x_batch, edge_index, edge_weight=edge_attr)
            
            # Guardar resultados
            previsoes.append(output.cpu().numpy())
            reais.append(y_batch.numpy())

    # Converter para arrays numpy
    previsoes = np.array(previsoes).reshape(-1, 207) # [Amostras, Sensores]
    reais = np.array(reais).reshape(-1, 207)

    # --- 4. DESNORMALIZAÇÃO (Voltar para km/h) ---
    # Usamos a média e std salvos no dataset
    mean, std = dataset_teste.mean, dataset_teste.std
    previsoes_kmh = (previsoes * std) + mean
    reais_kmh = (reais * std) + mean

    # --- 5. MÉTRICAS ACADÊMICAS ---
    mae = np.mean(np.abs(reais_kmh - previsoes_kmh))
    rmse = np.sqrt(np.mean((reais_kmh - previsoes_kmh)**2))
    print(f"\n📈 RESULTADOS FINAIS (METR-LA):")
    print(f"   > MAE: {mae:.2f} km/h")
    print(f"   > RMSE: {rmse:.2f} km/h")

    # --- 6. VISUALIZAÇÃO DE UM SENSOR ESPECÍFICO ---
    sensor_id = 10 # Vamos olhar o sensor número 10
    plt.figure(figsize=(12, 5))
    plt.plot(reais_kmh[:100, sensor_id], label='Real (Velocidade)', color='blue', alpha=0.6)
    plt.plot(previsoes_kmh[:100, sensor_id], label='Predição ST-GCN', color='red', linestyle='--')
    plt.title(f'Predição de Tráfego - Sensor {sensor_id}')
    plt.xlabel('Tempo (Passos de 5 min)')
    plt.ylabel('Velocidade (km/h)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    avaliar_modelo()