from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import base64
from PIL import Image, ImageOps
import io
import tempfile
import shutil
from face_database import FaceDatabase

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

# データベースの初期化
try:
    db = FaceDatabase()
    print("Face database initialized successfully")
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
    return jsonify({
        'status': 'healthy',
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'db_initialized': db is not None
    })

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
            return jsonify({'error': 'データベースが初期化されていません'}), 500
        
        data = request.json
        filename = data.get('filename')
        label = data.get('label')
        
        if not filename or not label:
            return jsonify({'error': 'ファイル名とラベルが必要です'}), 400
        
        filepath = create_temp_image_path(filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'ファイルが見つかりません'}), 404
        
        db.add_face(filepath, label)
        return jsonify({'success': True, 'message': f'{label} を学習しました'})
        
    except Exception as e:
        print(f"Learn face error: {e}")
        return jsonify({'error': f'学習エラー: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict_face():
    try:
        if db is None:
            return jsonify({'error': 'データベースが初期化されていません'}), 500
        
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
        label_dir = os.path.join("known_faces", label)
        match_image_data = None
        
        if os.path.exists(label_dir):
            image_list = os.listdir(label_dir)
            if image_list:
                match_path = os.path.join(label_dir, image_list[0])
                match_image_data = image_to_base64(match_path)
        
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
            return jsonify({'error': 'データベースが初期化されていません'}), 500
        
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
            db.add_face(filepath, label)
            return jsonify({'success': True, 'message': f'{label} に追加しました'})
        else:
            # 間違っている場合は新しいラベルを使用
            new_label = data.get('new_label')
            if not new_label:
                return jsonify({'error': '新しいラベルが必要です'}), 400
            
            db.add_face(filepath, new_label)
            return jsonify({'success': True, 'message': f'{new_label} に追加しました'})
            
    except Exception as e:
        print(f"Confirm prediction error: {e}")
        return jsonify({'error': f'確認エラー: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'ファイルサイズが大きすぎます'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'サーバー内部エラーが発生しました'}), 500

if __name__ == '__main__':
    # 本番環境では自動的にgunicornが使用される
    app.run(debug=False, host='0.0.0.0', port=PORT)