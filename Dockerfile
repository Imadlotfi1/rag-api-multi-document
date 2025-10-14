FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Indiquer à Hugging Face sur quel port notre application écoute
EXPOSE 7860

# Lancer l'application sur le port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
