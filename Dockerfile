# Image base légère et fiable
FROM python:3.11-slim

# Bonnes pratiques + limiter le multi-thread (OpenBLAS/NumPy)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# Dépendances système minimales (ok pour opencv-python-headless)
# libgl1 est inutile en headless, on garde seulement libglib2.0-0
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les deps Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier uniquement ce qui est nécessaire pour exécuter le job
COPY preopt_pdf_sharp_petitfinmarche.py /app/
COPY runner.py /app/

# Cloud Run Jobs concatènera les --args après l'ENTRYPOINT
ENTRYPOINT ["python", "/app/runner.py"]
