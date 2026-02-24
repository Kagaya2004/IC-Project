import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader_STGCN import preparar_loaders

def treinar_modelo():
    # 1. Configurações
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Iniciando Treino na GPU: {torch.cuda.get_device_name(0)}")
    
    epochs = 50
    batch_size = 32
    learning_rate = 0.001
    save_path = r'C:\IC\Projeto_V2\src\modelos\stgcn_metrla.pth'

    # 2. Carregar Dados do METR-LA
    # O loader agora retorna (X: [B, 12, 207, 1], Y: [B, 207])
    train_loader, test_loader = preparar_loaders(batch_size=batch_size)
    
    # 3. Inicializar Modelo
    # METR-LA básico usa 1 feature (velocidade)
    model = STGCN_TrafficModel(num_features=1).to(device)

    # 4. Carregar Grafo (Arestas e Pesos)
    data_dir = r'C:\IC\Projeto_V2\data\tratados'
    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt')).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr.pt')).to(device)

    # 5. Otimização
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 6. Loop de Treino
    print(f"Treinando em {len(train_loader.dataset)} amostras temporais...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Predição: Baseada nas últimas 12 observações
            output = model(x_batch, edge_index, edge_weight=edge_attr)
            
            # Perda calculada sobre a velocidade de todos os sensores
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Validação simples
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                pred = model(x_val, edge_index, edge_weight=edge_attr)
                val_loss += criterion(pred, y_val).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
        
        print(f"Época [{epoch+1}/{epochs}] | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

    # Salvar o modelo treinado
    torch.save(model.state_dict(), save_path)
    print(f"✅ Modelo salvo em: {save_path}")

if __name__ == "__main__":
    treinar_modelo()