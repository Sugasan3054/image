# Enhanced Face Recognition App

エンハンスド顔認識アプリケーションは、機械学習を使用して顔を学習・認識するWebアプリケーションです。複数のクラウドストレージに対応し、PWA（Progressive Web App）として動作します。

## 🌟 主な機能

- **顔の学習**: 画像をアップロードして顔を学習
- **顔の認識**: アップロードした画像から学習済みの顔を認識
- **リアルタイム同期**: 複数のクラウドストレージとの自動同期
- **PWA対応**: モバイルデバイスでのアプリライクな体験
- **バックアップ機能**: 手動バックアップとデータ復旧
- **統計情報**: 学習済み顔の統計とシステム状態

## 🚀 対応ストレージ

- **PostgreSQL**: 本格的なデータベース運用
- **Amazon S3**: クラウドストレージ
- **MongoDB**: NoSQLデータベース
- **GitHub Gist**: 軽量なクラウドストレージ

## 📋 必要条件

### Python依存関係
```
Flask
Pillow
face_recognition
numpy
requests
werkzeug
```

### システム要件
- Python 3.7以上
- dlib (face_recognitionの依存関係)
- OpenCV (推奨)

## 🔧 インストール

1. **リポジトリのクローン**
```bash
git clone <repository-url>
cd enhanced-face-recognition-app
```

2. **依存関係のインストール**
```bash
pip install -r requirements.txt
```

3. **face_database.pyの準備**
```bash
# face_database.pyファイルが必要です
# EnhancedCloudFaceDatabaseクラスを実装してください
```

## ⚙️ 環境変数設定

### PostgreSQL設定
```bash
export DATABASE_URL="postgresql://username:password@host:port/database"
```

### Amazon S3設定
```bash
export S3_BUCKET_NAME="your-bucket-name"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
```

### MongoDB設定
```bash
export MONGODB_URI="mongodb://username:password@host:port/database"
```

### GitHub Gist設定
```bash
export GITHUB_TOKEN="your-github-token"
export GIST_ID="your-gist-id"
```

### その他の設定
```bash
export PORT=5000  # デフォルト: 5000
```

## 🚀 起動方法

### 開発環境
```bash
python app.py
```

### 本番環境（Gunicorn）
```bash
gunicorn app:app --bind 0.0.0.0:5000
```

### Docker（推奨）
```bash
docker build -t face-recognition-app .
docker run -p 5000:5000 --env-file .env face-recognition-app
```

## 📱 使用方法

### 1. 顔の学習
1. メイン画面で「画像をアップロード」
2. 学習したい顔の画像を選択
3. 「学習」ボタンをクリック
4. 人物の名前（ラベル）を入力
5. 「学習実行」で完了

### 2. 顔の認識
1. 認識したい画像をアップロード
2. 「認識」ボタンをクリック
3. 結果が表示される
4. 必要に応じて「確認」で学習データを改善

### 3. 統計情報の確認
- `/stats` エンドポイントで統計情報を取得
- 学習済み顔の数、同期状態などを確認

## 🔌 API エンドポイント

### メイン機能
- `POST /upload` - 画像のアップロード
- `POST /learn` - 顔の学習
- `POST /predict` - 顔の認識
- `POST /confirm` - 認識結果の確認・修正

### 管理機能
- `GET /health` - ヘルスチェック
- `GET /stats` - 統計情報
- `POST /sync` - 手動同期
- `POST /backup` - バックアップ作成
- `GET /labels` - 学習済みラベル一覧
- `GET /storage-info` - ストレージ情報

### PWA機能
- `GET /manifest.json` - PWAマニフェスト
- `GET /sw.js` - サービスワーカー

## 🛡️ セキュリティ機能

- **ファイル検証**: 許可された画像形式のみ受け入れ
- **サイズ制限**: 最大16MBまでのファイルサイズ
- **セキュアファイル名**: werkzeugによる安全なファイル名生成
- **一時ファイル管理**: アップロードファイルの自動クリーンアップ

## 🔄 同期機能

- **自動同期**: 設定された間隔（デフォルト300秒）で自動同期
- **フォールバック**: プライマリストレージ障害時の自動切り替え
- **手動同期**: `/sync` エンドポイントによる即座の同期
- **バックアップ**: データ保護のための定期バックアップ

## 📊 監視・ログ

### ログレベル
- `INFO`: 一般的な動作ログ
- `ERROR`: エラーレベルのログ
- `WARNING`: 警告レベルのログ

### ヘルスチェック
`GET /health` エンドポイントで以下を確認：
- データベース接続状態
- ストレージ設定
- アップロードフォルダ状態
- 環境変数設定

## 🚀 デプロイ

### Railway
```bash
# railway.tomlで設定
railway up
```

### Heroku
```bash
# Procfileで設定
heroku create your-app-name
git push heroku main
```

### AWS/GCP
- Docker化してコンテナサービスにデプロイ
- 環境変数を適切に設定

## 🔧 トラブルシューティング

### よくある問題

1. **顔が検出されない**
   - 画像が鮮明であることを確認
   - 顔が正面を向いていることを確認
   - 照明が適切であることを確認

2. **データベース接続エラー**
   - 環境変数が正しく設定されているか確認
   - ネットワーク接続を確認
   - 認証情報を確認

3. **同期エラー**
   - `/sync` エンドポイントで手動同期を試行
   - ストレージサービスの状態を確認
   - フォールバックストレージの設定を確認

### ログの確認
```bash
# アプリケーションログ
tail -f app.log

# システムログ
journalctl -u face-recognition-app
```

## 🤝 貢献

1. フォークを作成
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチをプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は `LICENSE` ファイルをご覧ください。

## 🙏 謝辞

- [face_recognition](https://github.com/ageitgey/face_recognition) - 顔認識ライブラリ
- [Flask](https://flask.palletsprojects.com/) - Webフレームワーク
- [Pillow](https://pillow.readthedocs.io/) - 画像処理ライブラリ

## 📞 サポート

問題や質問がある場合は、以下の方法でお問い合わせください：

- GitHub Issues: [プロジェクトのIssues]
- メール: [your-email@example.com]
- ドキュメント: [プロジェクトWiki]

---

**注意**: このアプリケーションは顔認識技術を使用しています。プライバシーと個人情報保護に関する法律を遵守して使用してください。