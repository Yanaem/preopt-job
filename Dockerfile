FROM python:3.11-slim

# Libs système utiles à OpenCV (headless) et PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du script utilisateur + runner
COPY . .

# Commande par défaut: le wrapper
CMD ["python", "runner.py"]
