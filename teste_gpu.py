import time
print("1. Iniciando carregamento do Torch (isso pode demorar minutos na primeira vez)...")
start = time.time()

try:
    import torch
    print(f"2. Torch carregado em {time.time() - start:.2f} segundos.")
    
    print(f"3. Versão do Torch: {torch.__version__}")
    print(f"4. CUDA disponível: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"5. Nome da GPU: {torch.cuda.get_device_name(0)}")
        # Teste real de memória
        x = torch.rand(1, device='cuda')
        print("6. Teste de tensor na GPU: Sucesso!")
except Exception as e:
    print(f"Erro detectado: {e}")

print("7. Fim do teste.")