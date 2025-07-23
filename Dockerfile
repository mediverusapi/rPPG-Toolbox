# ------------------------------------------------------
# Stage 1 – build a slim Python runtime with all deps
# ------------------------------------------------------
FROM python:3.10-slim AS base

# Prevents Python from writing .pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install OS packages needed by opencv-python wheels and numpy compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libopenblas-dev \
        liblapack-dev \
        gfortran \
        pkg-config \
        && rm -rf /var/lib/apt/lists/*

# Set work dir
WORKDIR /app

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Copy dependency list & install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------
# Stage 2 – copy source code (keeps rebuilds fast)
# ------------------------------------------------------
FROM base AS final

WORKDIR /app
COPY . .

EXPOSE 8000

# Gunicorn isn't needed; Uvicorn is fine for a single-process container.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    