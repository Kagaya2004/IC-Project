Este projeto realiza uma análise comparativa entre as redes neurais Spatio-Temporal Graph Convolutional Network (STGCN) e Diffusion Convolutional Recurrent Neural Network (DCRNN) para a predição de tráfego.

📊 Sobre o Dataset (METR-LA)
Localização: Rodovias de Los Angeles.

Período de Coleta: De 1 de março de 2012 a 30 de junho de 2012.

Duração: 4 meses de dados contínuos.

Granularidade: Agregado em intervalos de 5 minutos (288 amostras/dia por sensor).

Sensores: 207 detectores de laço monitorados.

🚀 Ordem de Execução
Siga rigorosamente a ordem abaixo para garantir a integridade do pipeline de dados:

1. Pré-Processamento
Prepara e limpa os dados brutos para os modelos.

python -m src.preprocessamento.dataCleaning: Trata valores ausentes via interpolação linear e propagação.

python -m src.preprocessamento.featureEngineering: Adiciona features temporais (seno/cosseno de horário e dia).

python -m src.preprocessamento.preparar_grafo: Gera as matrizes de adjacência (ST-GCN) e de difusão (DCRNN).

2. Definição e Treinamento
Os arquivos modelo_STGCN.py e modelo_DCRNN.py definem as arquiteturas. Execute os treinos abaixo:

python -m src.treinamento.treinar_STGCN: Treino focado em predição de curto prazo (t+1).

python -m src.treinamento.treinar_DCRNN: Treino focado no horizonte completo de 60 minutos (12 passos).

3. Avaliação e Resultados
Geração de evidências estatísticas e visuais para o relatório de IC.

python -m src.utilidades.avaliar_modelos: Gera métricas comparativas (MAE, RMSE, Latência).

python -m src.utilidades.erro_sensor: Cria o scatter plot de erro por sensor.

python -m src.utilidades.visualizar_horizonte: Plota a curva de degradação do erro ao longo de 1 hora.

💻 Hardware de Referência
GPU: NVIDIA GeForce RTX 3050 (6GB VRAM).

Framework: PyTorch.

💡 Dicas de Execução
Sempre execute os comandos a partir da raiz do projeto.

Os resultados (pesos e logs) são salvos em src/modelos/ e data/logs/.