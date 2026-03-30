import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import time
import pandas as pd
from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader_Traffic import preparar_loaders

def treinar_stgcn_multi_step():
    # Configurações de Hardware e Ambiente
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Identificação única do experimento para organização da IC
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"treino_stgcn_{timestamp}.txt")
    
    def log_print(msg):
        print(msg)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(msg + "\n")

    # 1. Hiperparâmetros (Horizonte de 12 passos = 60 minutos)
    batch_size = 32
    epochs = 50
    lr = 0.001
    
    # Preparação dos dados: 1 hora de histórico para prever 1 hora à frente
    train_loader, test_loader = preparar_loaders(batch_size=batch_size, window_size=12, horizon=12)
    
    # Instanciação da ST-GCN (Multi-step nativa)
    model = STGCN_TrafficModel(num_features=5).to(device)
    
    # Carregamento do Grafo (Agora usando os pesos NORMALIZADOS SIMETRICAMENTE)
    try:
        edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
        edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)
        log_print(f"✅ Grafo carregado com {edge_index.shape[1]} arestas normalizadas.")
    except FileNotFoundError:
        log_print("❌ Erro: Arquivos de grafo não encontrados. Certifique-se de executar o script de preparação do grafo primeiro.")
        return

    # Otimização e Critério de Perda
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # MAE mascareado para ignorar sensores falhos (velocidade = 0)
    mae_criterion = nn.L1Loss(reduction='none')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    log_print(f"📊 [ST-GCN] Iniciando Treinamento (60 min)")
    log_print(f"💻 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    best_val_mae = float('inf')
    history = []
    start_train = time.time()

    # 2. Loop de Treinamento
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device) # y shape: [Batch, 12, Nodes]
            
            optimizer.zero_grad()
            # Predição simultânea de todos os 12 passos
            output = model(x, edge_index, edge_weight=edge_attr) 
            
            # Cálculo da perda apenas onde os dados são válidos
            mask = (y > 0).float()
            loss = (mae_criterion(output, y) * mask).sum() / (mask.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 3. Validação (Conjunto de Teste)
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for xv, yv in test_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = model(xv, edge_index, edge_weight=edge_attr)
                
                mask_v = (yv > 0).float()
                v_mae = (mae_criterion(pred, yv) * mask_v).sum() / (mask_v.sum() + 1e-8)
                val_mae += v_mae.item()

        avg_train_mae = train_loss / len(train_loader)
        avg_val_mae = val_mae / len(test_loader)
        
        # Ajuste dinâmico da taxa de aprendizado
        scheduler.step(avg_val_mae)
        
        # Salvamento do Melhor Modelo (Checkpoint para avaliação final)
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), os.path.join(model_dir, 'stgcn_model.pth'))
            status = "⭐ Melhor Modelo!"
        else:
            status = ""

        log_print(f"Ep [{epoch+1:02d}/{epochs}] | Train MAE: {avg_train_mae:.4f} | Val MAE: {avg_val_mae:.4f} {status}")
        history.append({"epoch": epoch+1, "train_mae": avg_train_mae, "val_mae": avg_val_mae})

    # Finalização e Salvamento do Histórico para Gráficos
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_train)))
    log_print(f"\n🏆 Treino ST-GCN Finalizado em {total_time}")
    
    # Nome do CSV compatível com os scripts de avaliação automatizada
    history_path = os.path.join(log_dir, f"history_stgcn_{timestamp}.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    log_print(f"📄 Histórico salvo em: {history_path}")

if __name__ == "__main__":
    treinar_stgcn_multi_step()