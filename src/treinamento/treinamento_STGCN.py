import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import pandas as pd
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader_STGCN import preparar_loaders

def treinar_modelo():
    # 1. Configurações de Hardware e Caminhos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)

    # 2. Configuração do Log
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"treino_log_{timestamp}.txt")
    history_csv = os.path.join(log_dir, f"curva_aprendizado_{timestamp}.csv")
    
    def log_and_print(msg):
        print(msg)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(msg + "\n")

    log_and_print(f"🚀 Iniciando Treino: {datetime.datetime.now()}")
    log_and_print(f"💻 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 3. Carregar Dados
    batch_size = 32
    epochs = 50
    train_loader, test_loader = preparar_loaders(batch_size=batch_size)
    
    # 4. Inicializar Modelo e Grafo
    # METR-LA usa 1 feature (velocidade)
    model = STGCN_TrafficModel(num_features=1).to(device)
    edge_index = torch.load(os.path.join(data_dir, 'edge_index.pt'), weights_only=True).to(device)
    edge_attr = torch.load(os.path.join(data_dir, 'edge_attr.pt'), weights_only=True).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Loop de Treinamento
    history = []
    log_and_print(f"📊 Amostras: {len(train_loader.dataset)} | Épocas: {epochs}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch, edge_index, edge_weight=edge_attr)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validação
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                pred = model(x_val, edge_index, edge_weight=edge_attr)
                val_loss += criterion(pred, y_val).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
        
        epoch_msg = f"Época [{epoch+1}/{epochs}] | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}"
        log_and_print(epoch_msg)
        history.append({"epoch": epoch+1, "train_loss": avg_train, "val_loss": avg_val})

    # 6. Finalização e Salvamento
    pd.DataFrame(history).to_csv(history_csv, index=False)
    model_path = os.path.join(model_dir, 'stgcn_metrla.pth')
    torch.save(model.state_dict(), model_path)
    
    log_and_print(f"✅ Treino concluído às {datetime.datetime.now()}")
    log_and_print(f"📦 Modelo salvo em: {model_path}")
    log_and_print(f"📝 Histórico salvo em: {history_csv}")

if __name__ == "__main__":
    treinar_modelo()