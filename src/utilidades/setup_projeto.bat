@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   CONFIGURADOR AUTOMATICO: PROJETO V2 (PYTHON 3.12.8)
echo ========================================================

:: 1. INSTALAR PYTHON 3.12.8
echo [1/4] Verificando instalacao do Python 3.12.8...
python --version 2>NUL | findstr /R "3.12.8" >NUL
if %errorlevel% neq 0 (
    echo [!] Python 3.12.8 nao encontrado. Instalando via winget...
    winget install -e --id Python.Python.3.12 --version 3.12.8 --scope machine
    echo [OK] Por favor, reinicie este script apos a instalacao terminar.
    pause
    exit
) else (
    echo [OK] Python 3.12.8 ja esta instalado.
)

:: 2. CRIAR AMBIENTE VIRTUAL (VENV)
echo.
echo [2/4] Criando ambiente virtual (.venv)...
if not exist ".venv" (
    python -m venv .venv
    echo [OK] Ambiente virtual criado com sucesso.
) else (
    echo [INFO] Pasta .venv ja existe. Pulando criacao.
)

:: 3. GERAR ARQUIVO REQUIREMENTS.TXT
echo.
echo [3/4] Gerando arquivo de requisitos atualizado...
(
echo aiohappyeyeballs==2.6.1
echo aiohttp==3.13.3
echo aiosignal==1.4.0
echo attrs==25.4.0
echo blosc2==4.1.0
echo certifi==2026.1.4
echo charset-normalizer==3.4.4
echo colorama==0.4.6
echo contourpy==1.3.3
echo cycler==0.12.1
echo et_xmlfile==2.0.0
echo filelock==3.20.3
echo fonttools==4.61.1
echo frozenlist==1.8.0
echo fsspec==2026.1.0
echo geographiclib==2.1
echo geopy==2.4.1
echo h5py==3.15.1
echo idna==3.11
echo Jinja2==3.1.6
echo joblib==1.5.3
echo kiwisolver==1.4.9
echo MarkupSafe==3.0.3
echo matplotlib==3.10.8
echo mpmath==1.3.0
echo msgpack==1.1.2
echo multidict==6.7.0
echo ndindex==1.10.1
echo networkx==3.6.1
echo numexpr==2.14.1
echo numpy==1.26.4
echo openpyxl==3.1.5
echo packaging==25.0
echo pandas==2.1.4
echo pillow==12.1.0
echo propcache==0.4.1
echo psutil==7.2.1
echo py-cpuinfo==9.0.0
echo pyparsing==3.3.1
echo python-dateutil==2.9.0.post0
echo pytz==2025.2
echo PyYAML==6.0.3
echo requests==2.32.5
echo scikit-learn==1.8.0
echo scipy==1.17.0
echo seaborn==0.13.2
echo setuptools==80.9.0
echo six==1.17.0
echo sympy==1.13.1
echo tables==3.11.1
echo threadpoolctl==3.6.0
echo torch==2.5.1+cu121
echo torch-geometric==2.7.0
echo torch_cluster==1.6.3+pt25cu121
echo torch_scatter==2.1.2+pt25cu121
echo torch_sparse==0.6.18+pt25cu121
echo torch_spline_conv==1.2.2+pt25cu121
echo torchaudio==2.5.1+cu121
echo torchvision==0.20.1+cu121
echo tqdm==4.67.1
echo typing_extensions==4.15.0
echo tzdata==2025.3
echo urllib3==2.6.3
echo xxhash==3.6.0
echo yarl==1.22.0
) > requirements.txt
echo [OK] requirements.txt criado.

:: 4. ATIVAR VENV E INSTALAR BIBLIOTECAS
echo.
echo [4/4] Ativando ambiente e instalando bibliotecas...
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

echo.
echo ========================================================
echo   PROCESSO CONCLUIDO! SEU AMBIENTE ESTA PRONTO.
echo ========================================================
pause