from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import base64
from PIL import Image, ImageOps
import io
from face_database import FaceDatabase

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# アップロードフォルダの作成
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# データベースの初期化
db = FaceDatabase()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    """画像をBase64エンコードして返す"""
    try:
        with Image.open(image_path) as img:
            img = ImageOps.fit(img, (250, 250), method=Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_data = buffer.getvalue()
            return base64.b64encode(img_data).decode('utf-8')
    except Exception as e:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'ファイルが選択されていません'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'ファイルが選択されていません'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 画像をBase64エンコードして返す
        img_base64 = image_to_base64(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'image_data': img_base64
        })
    
    return jsonify({'error': '無効なファイル形式です'}), 400

@app.route('/learn', methods=['POST'])
def learn_face():
    data = request.json
    filename = data.get('filename')
    label = data.get('label')
    
    if not filename or not label:
        return jsonify({'error': 'ファイル名とラベルが必要です'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'ファイルが見つかりません'}), 404
    
    try:
        db.add_face(filepath, label)
        return jsonify({'success': True, 'message': f'{label} を学習しました'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_face():
    data = request.json
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'ファイル名が必要です'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'ファイルが見つかりません'}), 404
    
    try:
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
        return jsonify({'error': str(e)}), 500

@app.route('/confirm', methods=['POST'])
def confirm_prediction():
    data = request.json
    filename = data.get('filename')
    label = data.get('label')
    is_correct = data.get('is_correct', True)
    
    if not filename:
        return jsonify({'error': 'ファイル名が必要です'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'ファイルが見つかりません'}), 404
    
    try:
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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 本番環境では自動的にgunicornが使用される
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)