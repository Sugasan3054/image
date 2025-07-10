FROM python:3.9-slim

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# requirements.txt をコピーして依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルをコピー
COPY . .

# 必要なディレクトリを作成
RUN mkdir -p uploads known_faces

# ポートを公開
EXPOSE 5000

# アプリケーションを起動
CMD ["python", "app.py"]