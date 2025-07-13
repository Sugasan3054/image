from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import base64
from PIL import Image, ImageOps
import io
import tempfile
import shutil
import requests
import json
import numpy as np
import face_recognition
from datetime import datetime

app = Flask(__name__)

# 環境変数からポートを取得
PORT = int(os.environ.get('PORT', 5000))

# 一時ディレクトリを使用（Railway対応）
TEMP_DIR = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = os.path.join(TEMP_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# アップロードフォルダの作成
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print(f"Upload folder created: {app.config['UPLOAD_FOLDER']}")
except Exception as e:
    print(f"Error creating upload folder: {e}")

# GitHub Gist Face Database クラス
class GitHubGistFaceDB:
    def __init__(self, github_token, gist_id=None):
        self.github_token = github_token
        self.gist_id = gist_id
        self.gist_filename = 'face_database.json'
        self.known_faces = []
        self.known_labels = []
        self.face_images = {}  # ラベルと画像データのマッピング
        
        # 既存データの読み込み
        self.load_known_faces()
    
    def load_known_faces(self):
        """GitHub Gist から顔データを読み込む"""
        if not self.gist_id:
            print("No gist_id provided, starting with empty database")
            return
        
        try:
            url = f"https://api.github.com/gists/{self.gist_id}"
            headers = {"Authorization": f"token {self.github_token}"}
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                gist_data = response.json()
                
                if self.gist_filename not in gist_data['files']:
                    print(f"File {self.gist_filename} not found in gist")
                    return
                
                content = gist_data['files'][self.gist_filename]['content']
                data = json.loads(content)
                
                for item in data:
                    encoding = np.array(item['encoding'])
                    label = item['label']
                    
                    self.known_faces.append(encoding)
                    self.known_labels.append(label)
                    
                    # 画像データがあれば保存
                    if 'image_data' in item:
                        if label not in self.face_images:
                            self.face_images[label] = []
                        self.face_images[label].append(item['image_data'])
                
                print(f"Loaded {len(self.known_faces)} known faces from GitHub Gist")
            else:
                print(f"Failed to load from GitHub: {response.status_code}")
                if response.status_code == 404:
                    print("Gist not found. Please check your gist_id.")
                elif response.status_code == 401:
                    print("Unauthorized. Please check your GitHub token.")
        except Exception as e:
            print(f"Error loading from GitHub: {e}")
    
    def save_to_gist(self):
        """データベースをGitHub Gistに保存"""
        try:
            # データを準備
            data = []
            for i, encoding in enumerate(self.known_faces):
                label = self.known_labels[i]
                item = {
                    'label': label,
                    'encoding': encoding.tolist(),
                    'created_at': datetime.now().isoformat()
                }
                
                # 画像データがあれば追加
                if label in self.face_images and self.face_images[label]:
                    item['image_data'] = self.face_images[label][0]  # 最初の画像を使用
                
                data.append(item)
            
            content = json.dumps(data, indent=2)
            
            headers = {"Authorization": f"token {self.github_token}"}
            
            if self.gist_id:
                # 既存のGistを更新
                url = f"https://api.github.com/gists/{self.gist_id}"
                payload = {
                    "files": {
                        self.gist_filename: {
                            "content": content
                        }
                    }
                }
                response = requests.patch(url, json=payload, headers=headers)
            else:
                # 新しいGistを作成
                url = "https://api.github.com/gists"
                payload = {
                    "description": "Face recognition database",
                    "public": False,
                    "files": {
                        self.gist_filename: {
                            "content": content
                        }
                    }
                }
                response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code in [200, 201]:
                if not self.gist_id:
                    self.gist_id = response.json()['id']
                    print(f"Created new gist: {self.gist_id}")
                    print(f"Gist URL: https://gist.github.com/{self.gist_id}")
                else:
                    print(f"Updated gist: {self.gist_id}")
                return True
            else:
                print(f"Failed to save to GitHub: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error saving to GitHub: {e}")
            return False
    
    def add_face(self, image_path, label):
        """新しい顔を学習データに追加"""
        try:
            # 画像を読み込み
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                raise ValueError("顔が検出されませんでした")
            
            # 複数の顔が検出された場合は最初の顔を使用
            if len(encodings) > 1:
                print(f"Warning: {len(encodings)} faces detected. Using the first one.")
            
            # 画像をBase64エンコード
            image_base64 = self.image_to_base64(image_path)
            
            # メモリ上のデータに追加
            self.known_faces.append(encodings[0])
            self.known_labels.append(label)
            
            # 画像データを保存
            if label not in self.face_images:
                self.face_images[label] = []
            if image_base64:
                self.face_images[label].append(image_base64)
            
            # Gistに保存
            success = self.save_to_gist()
            if success:
                print(f"Added face for label: {label}")
                return True
            else:
                # 失敗時はメモリからも削除
                self.known_faces.pop()
                self.known_labels.pop()
                if label in self.face_images and self.face_images[label]:
                    self.face_images[label].pop()
                return False
                
        except Exception as e:
            raise Exception(f"顔の追加に失敗しました: {str(e)}")
    
    def predict(self, image_path, threshold=0.6):
        """顔を予測"""
        try:
            # 入力画像から顔エンコーディングを取得
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                return None, 0, None
            
            # 学習済みの顔がない場合
            if not self.known_faces:
                return "unknown", 0, None
            
            # 顔を比較
            input_encoding = encodings[0]
            distances = face_recognition.face_distance(self.known_faces, input_encoding)
            
            # 最も近い顔を取得
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]
            
            # 類似度を計算
            similarity = 1 - min_distance
            
            if similarity >= threshold:
                predicted_label = self.known_labels[min_distance_index]
                return predicted_label, similarity, min_distance
            else:
                return "unknown", similarity, min_distance
                
        except Exception as e:
            raise Exception(f"顔の予測に失敗しました: {str(e)}")
    
    def image_to_base64(self, image_path):
        """画像をBase64エンコードして返す"""
        try:
            with Image.open(image_path) as img:
                # 画像サイズを制限してメモリ使用量を減らす
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                
                # RGBモードに変換
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_data = buffer.getvalue()
                return base64.b64encode(img_data).decode('utf-8')
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None
    
    def get_known_labels(self):
        """学習済みのラベル一覧を取得"""
        return list(set(self.known_labels))
    
    def get_face_count(self, label):
        """特定のラベルの顔の数を取得"""
        return self.known_labels.count(label)
    
    def get_stats(self):
        """データベースの統計情報を取得"""
        return {
            'total_faces': len(self.known_faces),
            'unique_labels': len(set(self.known_labels)),
            'labels': dict([(label, self.get_face_count(label)) for label in set(self.known_labels)]),
            'gist_id': self.gist_id
        }
    
    def get_face_image(self, label):
        """特定のラベルの顔画像を取得"""
        if label in self.face_images and self.face_images[label]:
            return self.face_images[label][0]
        return None

# 環境変数からGitHub設定を取得
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GIST_ID = os.environ.get('GIST_ID')

# データベースの初期化
try:
    if not GITHUB_TOKEN:
        print("Warning: GITHUB_TOKEN not set. Using empty database.")
        db = None
    else:
        db = GitHubGistFaceDB(GITHUB_TOKEN, GIST_ID)
        print("GitHub Gist Face database initialized successfully")
        print(f"Database stats: {db.get_stats()}")
except Exception as e:
    print(f"Error initializing face database: {e}")
    db = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    """画像をBase64エンコードして返す"""
    try:
        with Image.open(image_path) as img:
            # 画像サイズを制限してメモリ使用量を減らす
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            # RGBモードに変換（RGBA や P モードの場合）
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_data = buffer.getvalue()
            return base64.b64encode(img_data).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def create_temp_image_path(filename):
    """一時的な画像パスを作成"""
    return os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """ヘルスチェック用エンドポイント"""
    stats = db.get_stats() if db else {'error': 'Database not initialized'}
    return jsonify({
        'status': 'healthy',
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'db_initialized': db is not None,
        'github_token_set': bool(GITHUB_TOKEN),
        'gist_id': GIST_ID,
        'db_stats': stats
    })

@app.route('/stats')
def get_stats():
    """データベースの統計情報を取得"""
    if db is None:
        return jsonify({'error': 'データベースが初期化されていません'}), 500
    
    try:
        stats = db.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'統計情報の取得に失敗しました: {str(e)}'}), 500

# PWA用のmanifest.jsonとservice-worker.jsを提供
@app.route('/manifest.json')
def manifest():
    return jsonify({
        "name": "Face Recognition App",
        "short_name": "FaceApp",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#000000",
        "icons": [
            {
                "src": "/static/icon-192x192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "/static/icon-512x512.png",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    })

@app.route('/sw.js')
def service_worker():
    return send_file('static/sw.js', mimetype='application/javascript')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'ファイルが選択されていません'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'ファイルが選択されていません'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '無効なファイル形式です（PNG, JPG, JPEG, GIFのみ）'}), 400
        
        # ファイルサイズチェック
        file_data = file.read()
        if len(file_data) > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'ファイルサイズが大きすぎます（16MB以下）'}), 400
        
        # ファイルポインタを先頭に戻す
        file.seek(0)
        
        # セキュアなファイル名を生成
        filename = secure_filename(file.filename)
        filepath = create_temp_image_path(filename)
        
        # ファイルを保存
        file.save(filepath)
        
        # 保存されたファイルが存在するか確認
        if not os.path.exists(filepath):
            return jsonify({'error': 'ファイルの保存に失敗しました'}), 500
        
        # 画像をBase64エンコードして返す
        img_base64 = image_to_base64(filepath)
        
        if img_base64 is None:
            return jsonify({'error': '画像の処理に失敗しました'}), 500
        
        return jsonify({
            'success': True,
            'filename': filename,
            'image_data': img_base64,
            'file_size': len(file_data)
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': f'アップロードエラー: {str(e)}'}), 500

@app.route('/learn', methods=['POST'])
def learn_face():
    try:
        if db is None:
            return jsonify({'error': 'データベースが初期化されていません。GitHub Tokenを設定してください。'}), 500
        
        data = request.json
        filename = data.get('filename')
        label = data.get('label')
        
        if not filename or not label:
            return jsonify({'error': 'ファイル名とラベルが必要です'}), 400
        
        filepath = create_temp_image_path(filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'ファイルが見つかりません'}), 404
        
        success = db.add_face(filepath, label)
        
        if success:
            return jsonify({
                'success': True, 
                'message': f'{label} を学習しました',
                'gist_id': db.gist_id
            })
        else:
            return jsonify({'error': 'GitHub Gistへの保存に失敗しました'}), 500
        
    except Exception as e:
        print(f"Learn face error: {e}")
        return jsonify({'error': f'学習エラー: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict_face():
    try:
        if db is None:
            return jsonify({'error': 'データベースが初期化されていません。GitHub Tokenを設定してください。'}), 500
        
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'ファイル名が必要です'}), 400
        
        filepath = create_temp_image_path(filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'ファイルが見つかりません'}), 404
        
        label, similarity, _ = db.predict(filepath)
        
        if label is None:
            return jsonify({'error': '顔が検出されませんでした'}), 400
        
        # マッチした顔の画像を取得
        match_image_data = None
        if label != "unknown":
            match_image_data = db.get_face_image(label)
        
        return jsonify({
            'success': True,
            'label': label,
            'similarity': similarity,
            'similarity_percent': f"{similarity * 100:.2f}%",
            'match_image': match_image_data
        })
        
    except Exception as e:
        print(f"Predict face error: {e}")
        return jsonify({'error': f'予測エラー: {str(e)}'}), 500

@app.route('/confirm', methods=['POST'])
def confirm_prediction():
    try:
        if db is None:
            return jsonify({'error': 'データベースが初期化されていません。GitHub Tokenを設定してください。'}), 500
        
        data = request.json
        filename = data.get('filename')
        label = data.get('label')
        is_correct = data.get('is_correct', True)
        
        if not filename:
            return jsonify({'error': 'ファイル名が必要です'}), 400
        
        filepath = create_temp_image_path(filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'ファイルが見つかりません'}), 404
        
        if is_correct:
            # 正しい場合は既存のラベルに追加
            success = db.add_face(filepath, label)
            if success:
                return jsonify({
                    'success': True, 
                    'message': f'{label} に追加しました',
                    'gist_id': db.gist_id
                })
            else:
                return jsonify({'error': 'GitHub Gistへの保存に失敗しました'}), 500
        else:
            # 間違っている場合は新しいラベルを使用
            new_label = data.get('new_label')
            if not new_label:
                return jsonify({'error': '新しいラベルが必要です'}), 400
            
            success = db.add_face(filepath, new_label)
            if success:
                return jsonify({
                    'success': True, 
                    'message': f'{new_label} に追加しました',
                    'gist_id': db.gist_id
                })
            else:
                return jsonify({'error': 'GitHub Gistへの保存に失敗しました'}), 500
            
    except Exception as e:
        print(f"Confirm prediction error: {e}")
        return jsonify({'error': f'確認エラー: {str(e)}'}), 500

@app.route('/labels')
def get_labels():
    """学習済みのラベル一覧を取得"""
    if db is None:
        return jsonify({'error': 'データベースが初期化されていません'}), 500
    
    try:
        labels = db.get_known_labels()
        return jsonify({'labels': labels})
    except Exception as e:
        return jsonify({'error': f'ラベル取得エラー: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'ファイルサイズが大きすぎます'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'サーバー内部エラーが発生しました'}), 500

if __name__ == '__main__':
    # 本番環境では自動的にgunicornが使用される
    app.run(debug=False, host='0.0.0.0', port=PORT)