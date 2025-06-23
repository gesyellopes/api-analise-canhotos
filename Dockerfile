# Base image
FROM python:3.11-slim

# Define diretório de trabalho
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libzbar0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos de dependências Python
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código para o container
COPY . .

# Expor porta que a API utiliza
EXPOSE 5000

# Comando para rodar a API
CMD ["python", "api.py"]