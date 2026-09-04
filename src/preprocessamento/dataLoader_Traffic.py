import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TrafficDataset(Dataset):

  def __init__(
      self,
      data_array,
      window_size=12,
      horizon=12,
  ):
    # Recebe diretamente a matriz já processada e normalizada
    self.data = data_array
    self.window_size = window_size
    self.horizon = horizon

  def __len__(self):
    return len(self.data) - self.window_size - self.horizon + 1

  def __getitem__(self, idx):
    # X: [Window_Size, Nodes, Features] -> Entrada do modelo
    x = self.data[idx : idx + self.window_size]

    # y: [Horizon, Nodes] -> Alvo (apenas a velocidade, índice 0)
    y = self.data[
        idx + self.window_size : idx + self.window_size + self.horizon, :, 0
    ]

    return torch.tensor(x, dtype=torch.float32), torch.tensor(
        y, dtype=torch.float32
    )


def preparar_loaders(
    batch_size=32,
    window_size=12,
    horizon=12,
    train_ratio=0.8,
):
  base_path = r'C:\IC\Projeto_V2\data\tratados'
  file_path = os.path.join(base_path, 'METR-LA_featured.csv')

  print(f'Carregando base: {file_path}...')
  if not os.path.exists(file_path):
    raise FileNotFoundError(f'Erro: {file_path} não encontrado.')

  df = pd.read_csv(file_path, index_col=0)

  time_features_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
  speed_cols = [c for c in df.columns if c not in time_features_cols]

  v_raw = df[speed_cols].values.astype(np.float32)
  t_raw = df[time_features_cols].values.astype(np.float32)

  num_timesteps, num_nodes = v_raw.shape

  # 1. Ponto de corte cronológico (ex: 80% inicial para treino)
  split_idx = int(num_timesteps * train_ratio)

  v_train_raw = v_raw[:split_idx]
  v_test_raw = v_raw[split_idx:]

  t_train = t_raw[:split_idx]
  t_test = t_raw[split_idx:]

  # 2. Normalização calculada EXCLUSIVAMENTE nos dados de treino
  train_mean = v_train_raw.mean()
  train_std = v_train_raw.std()

  v_train_norm = (v_train_raw - train_mean) / (train_std + 1e-8)
  v_test_norm = (v_test_raw - train_mean) / (train_std + 1e-8)

  # Salve train_mean e train_std para desnormalizar as predições na avaliação:
  # np.save('scaler_params.npy', {'mean': train_mean, 'std': train_std})

  # 3. Montagem das dimensões [T, N, F] para Treino
  v_train_norm = np.expand_dims(v_train_norm, axis=-1)
  t_train_exp = np.tile(np.expand_dims(t_train, axis=1), (1, num_nodes, 1))
  data_train = np.concatenate([v_train_norm, t_train_exp], axis=-1).astype(
      np.float32
  )

  # 4. Montagem das dimensões [T, N, F] para Teste
  v_test_norm = np.expand_dims(v_test_norm, axis=-1)
  t_test_exp = np.tile(np.expand_dims(t_test, axis=1), (1, num_nodes, 1))
  data_test = np.concatenate([v_test_norm, t_test_exp], axis=-1).astype(
      np.float32
  )

  # Instanciação dos datasets sequenciais
  train_ds = TrafficDataset(data_train, window_size, horizon)
  test_ds = TrafficDataset(data_test, window_size, horizon)

  # Atenção: no treino podemos embaralhar a ordem dos batches (shuffle=True),
  # pois cada batch contém janelas que só olham para o próprio passado de treino.
  train_loader = DataLoader(
      train_ds, batch_size=batch_size, shuffle=True, pin_memory=True
  )
  test_loader = DataLoader(
      test_ds, batch_size=batch_size, shuffle=False, pin_memory=True
  )

  print(f'Divisão temporal realizada com sucesso!')
  print(f'Média de treino: {train_mean:.4f} | Desvio de treino: {train_std:.4f}')
  print(
      f'Treino: {len(train_ds)} amostras ({len(train_loader)} batches) | Teste:'
      f' {len(test_ds)} amostras ({len(test_loader)} batches)\n'
  )

  return train_loader, test_loader, train_mean, train_std