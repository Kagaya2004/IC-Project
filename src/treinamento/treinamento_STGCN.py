import torch
import torch.nn as nn
import torch.optim as optim
import os
# Ajuste do import para o seu novo nome de arquivo
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader_STGCN import preparar_loaders

def treinar_modelo():
    # 1. Configurações de Hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Iniciando Treino na GPU: {torch.cuda.get_device_name(0)}")

    # 2. Carregar Dados
    # O batch_size 32 é ideal para os 6GB da sua RTX 3050
    train_loader, test_loader = preparar_loaders(batch_size=32)
    
    # 3. Inicializar Modelo com 21 features
    # Carregamos uma amostra para confirmar o shape
    x_sample, _ = next(iter(train_loader))
    num_features = x_sample.shape[-1] # Deve ser 21
    
    model = STGCN_TrafficModel(num_features=num_features).to(device)
    
    # 4. Grafo Fixo (edge_index)
    edge_index = torch.load(r'C:\IC\Projeto_V2\data\tratados\edge_index.pt', weights_only=True).to(device)

    # 5. Perda (MSE) e Otimizador (Adam)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. Loop de Treinamento
    epochs = 50
    print(f"Amostras de treino: {len(train_loader.dataset)}")
    print("-" * 30)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward: O modelo processa espaço e tempo
            output = model(x_batch, edge_index)
            
            # Cálculo do erro na previsão de velocidade
            loss = criterion(output, y_batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Época [{epoch+1}/{epochs}] | Loss (MSE): {avg_loss:.6f}")

    # 7. Salvar o "Cérebro" Treinado
    save_path = r'C:\IC\Projeto_V2\src\modelo\stgcn_treinada.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("-" * 30)
    print(f"✅ Treino Finalizado! Modelo salvo em: {save_path}")

if __name__ == "__main__":
    treinar_modelo()