FROM python:3.9-slim

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# 一時ディレクトリを作成
RUN mkdir -p /tmp/uploads /tmp/known_faces

# ポートを公開
EXPOSE $PORT

# アプリケーションを実行
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --max-requests 1000 app:app