# 顔認識アプリケーション

このアプリケーションは、顔認識技術を使用して画像内の人物を識別・学習するWebアプリケーションです。

## 機能

- 🖼️ 画像アップロード
- 🎯 顔検出と認識
- 📚 顔データの学習
- 🔍 顔予測とマッチング
- ✅ 予測結果の確認と修正

## 必要な環境

- Python 3.8以上
- pip
- カメラ（オプション）

## セットアップ

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/face-recognition-app.git
cd face-recognition-app
```

### 2. 仮想環境の作成（推奨）

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 4. 必要なフォルダの作成

```bash
mkdir -p uploads known_faces
```

## 使用方法

### アプリケーションの起動

```bash
python app.py
```

ブラウザで `http://localhost:5000` にアクセスしてください。

### 基本的な使い方

1. **画像をアップロード**: 顔が含まれた画像をアップロードします
2. **顔を学習**: 新しい人物の場合は名前を入力して学習させます
3. **顔を予測**: 学習済みの顔データから最も類似した人物を予測します
4. **結果を確認**: 予測結果が正しければ確認、間違っていれば修正します

## Dockerでの実行（オプション）

```bash
# イメージのビルド
docker-compose build

# コンテナの起動
docker-compose up
```

## トラブルシューティング

### よくある問題

1. **dlib のインストールエラー**
   - Windows: Visual Studio Build Tools が必要
   - macOS: `brew install cmake` を実行
   - Linux: `apt-get install cmake build-essential`

2. **face-recognition のインストールエラー**
   - 最新のpipを使用: `pip install --upgrade pip`
   - 個別インストール: `pip install face-recognition --no-cache-dir`

3. **メモリ不足**
   - 大きな画像は自動的にリサイズされますが、メモリ不足の場合は画像サイズを小さくしてください

## 技術スタック

- **バックエンド**: Flask (Python)
- **フロントエンド**: HTML, CSS, JavaScript
- **顔認識**: face-recognition library (dlib)
- **画像処理**: Pillow (PIL)
- **機械学習**: scikit-learn

## ライセンス

MIT License

## 貢献

プルリクエストや課題報告を歓迎します。

## 注意事項

- このアプリケーションは教育・研究目的で作成されています
- 商用利用の場合は適切なライセンスを確認してください
- 個人情報の取り扱いには十分注意してください