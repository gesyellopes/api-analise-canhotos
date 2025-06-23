# Use uma imagem oficial do Python como base
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos do projeto para o container
COPY . /app

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta que o Flask usa
EXPOSE 5000

# Comando para rodar a aplicação
CMD ["python", "app.py"]