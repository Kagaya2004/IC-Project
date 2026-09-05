import os
import time
import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.modelos.modelo_STGCN import STGCN_TrafficModel
from src.preprocessamento.dataLoader_Traffic import preparar_loaders

def treinar_stgcn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = r'C:\IC\Projeto_V2'
    data_dir = os.path.join(base_dir, 'data', 'tratados')
    log_dir = os.path.join(base_dir, 'data', 'logs')
    model_dir = os.path.join(base_dir, 'src', 'modelos')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"treino_stgcn_{timestamp}.txt")
    
    def log_print(msg):
        print(msg)
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(msg + "\n")

    batch_size = 32
    epochs = 50
    lr = 0.001

    # 1. Carregamento dos dados sem data leakage (4 retornos)
    train_loader, test_loader, train_mean, train_std = preparar_loaders(
        batch_size=batch_size, window_size=12, horizon=12
    )

    # Persistência dos parâmetros de normalização
    scaler_path = os.path.join(data_dir, 'scaler_params.pt')
    torch.save({'mean': float(train_mean), 'std': float(train_std)}, scaler_path)
    log_print(f"Scaler salvo em {scaler_path} (Média: {train_mean:.4f}, Desvio: {train_std:.4f})")

    # Limiar para ignorar dados nulos no domínio normalizado
    zero_threshold = -train_mean / (train_std + 1e-8)

    # 2. Instanciação do Modelo
    model = STGCN_TrafficModel(num_nodes=207, num_features=5).to(device)

    # Carregamento do Grafo
    try:
        edge_index = torch.load(os.path.join(data_dir, 'edge_index_stgcn.pt'), weights_only=True).to(device)
        edge_attr = torch.load(os.path.join(data_dir, 'edge_attr_stgcn.pt'), weights_only=True).to(device)
        log_print(f"Grafo carregado com {edge_index.shape[1]} arestas.")
    except FileNotFoundError:
        log_print("Erro: Arquivos de grafo não encontrados. Execute preparar_grafo.py primeiro!")
        return

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mae_criterion = nn.L1Loss(reduction='none')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    log_print(f"GPU Detectada: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Caminhos para salvamento isolado
    caminho_pesos_salvar = os.path.join(model_dir, 'stgcn_model.pth')
    caminho_checkpoint = os.path.join(model_dir, 'stgcn_checkpoint.pth')

    start_epoch = 0
    best_val_mae = float('inf')
    history = []

    # 3. LÓGICA DE RETOMADA (FULL CHECKPOINTING)
    if os.path.exists(caminho_checkpoint):
        log_print(f"\n🔄 ARQUIVO DE CHECKPOINT DETECTADO!")
        log_print("Restaurando estado de execução do STGCN...")
        checkpoint = torch.load(caminho_checkpoint, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_mae = checkpoint['best_val_mae']
        history = checkpoint.get('history', [])
        
        log_print(f"-> Treinamento retomado a partir da Época {start_epoch+1}.")
        log_print(f"-> Melhor MAE anterior: {best_val_mae:.4f}\n")
    else:
        log_print("[STGCN] Iniciando Treinamento do zero...")

    start_train = time.time()

    # 4. Loop de Treinamento Seguro
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            train_loss = 0.0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                output = model(x, edge_index, edge_weight=edge_attr)

                # Máscara adaptada ao limiar normalizado
                mask = (y > zero_threshold).float()
                loss = (mae_criterion(output, y) * mask).sum() / (mask.sum() + 1e-8)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validação
            model.eval()
            val_mae = 0.0
            with torch.no_grad():
                for xv, yv in test_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    pred = model(xv, edge_index, edge_weight=edge_attr)

                    mask_v = (yv > zero_threshold).float()
                    v_mae = (mae_criterion(pred, yv) * mask_v).sum() / (mask_v.sum() + 1e-8)
                    val_mae += v_mae.item()

            avg_train_mae = train_loss / len(train_loader)
            avg_val_mae = val_mae / len(test_loader)
            scheduler.step(avg_val_mae)

            history.append({"epoch": epoch+1, "train_mae": avg_train_mae, "val_mae": avg_val_mae})

            # Salva Pesos Limpos para Inferência (Recorde)
            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                torch.save(model.state_dict(), caminho_pesos_salvar)
                status = "Melhor Modelo!"
            else:
                status = ""

            # Salva o Estado Completo para Resiliência Sistêmica (Sempre)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_mae': best_val_mae,
                'history': history
            }, caminho_checkpoint)

            current_lr = optimizer.param_groups[0]['lr']
            log_print(f"Ep [{epoch+1:02d}/{epochs}] | Train MAE: {avg_train_mae:.4f} | Val MAE: {avg_val_mae:.4f} | LR: {current_lr:.6f} {status}")

    except KeyboardInterrupt:
        log_print("\n\n⚠️ Treinamento INTERROMPIDO manualmente pelo usuário (Ctrl+C).")
        log_print("O progresso da última época concluída foi salvo integralmente no checkpoint.")

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_train)))
    
    log_print("\n" + "="*50)
    log_print(f"Treino ST-GCN Finalizado em {total_time}")
    log_print(f"Melhor MAE normalizado no Teste (Horizonte 12): {best_val_mae:.4f}")
    log_print("="*50)

    history_path = os.path.join(log_dir, f"history_stgcn_{timestamp}.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    log_print(f"Histórico salvo em: {history_path}")

if __name__ == "__main__":
    treinar_stgcn()