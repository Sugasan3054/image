FROM python:3.11-slim

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgtk-3-dev \
    libboost-all-dev \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 最初にdlibをインストール（時間がかかるため）
RUN pip install --no-cache-dir dlib

# requirements.txtをコピー（dlibを除く）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルをコピー
COPY . .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]