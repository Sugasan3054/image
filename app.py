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
import logging

# Import the enhanced face database
from face_database import EnhancedCloudFaceDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Environment variables
PORT = int(os.environ.get('PORT', 5000))

# Temporary directory for Railway compatibility
TEMP_DIR = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = os.path.join(TEMP_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    logger.info(f"Upload folder created: {app.config['UPLOAD_FOLDER']}")
except Exception as e:
    logger.error(f"Error creating upload folder: {e}")

# Configuration helper functions
def get_database_config():
    """Get database configuration from environment variables"""
    # Check for GitHub configuration
    github_token = os.environ.get('GITHUB_TOKEN')
    gist_id = os.environ.get('GIST_ID')
    
    # Check for PostgreSQL configuration
    database_url = os.environ.get('DATABASE_URL')
    
    # Check for S3 configuration
    s3_bucket = os.environ.get('S3_BUCKET_NAME')
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    
    # Check for MongoDB configuration
    mongo_url = os.environ.get('MONGODB_URI')
    
    # Determine primary storage type
    if database_url:
        primary_storage = 'postgresql'
        primary_config = {
            'database_url': database_url,
            'auto_sync': True,
            'sync_interval': 300
        }
    elif s3_bucket and aws_access_key and aws_secret_key:
        primary_storage = 's3'
        primary_config = {
            'bucket_name': s3_bucket,
            'aws_access_key_id': aws_access_key,
            'aws_secret_access_key': aws_secret_key,
            'auto_sync': True,
            'sync_interval': 300
        }
    elif mongo_url:
        primary_storage = 'mongodb'
        primary_config = {
            'mongo_url': mongo_url,
            'auto_sync': True,
            'sync_interval': 300
        }
    elif github_token:
        primary_storage = 'github'
        primary_config = {
            'github_token': github_token,
            'gist_id': gist_id,
            'auto_sync': True,
            'sync_interval': 300
        }
    else:
        logger.warning("No storage configuration found")
        return None, None
    
    # Configure fallback storage
    fallback_config = None
    if primary_storage != 'github' and github_token:
        fallback_config = {
            'type': 'github',
            'github_token': github_token,
            'gist_id': gist_id
        }
    elif primary_storage != 'postgresql' and database_url:
        fallback_config = {
            'type': 'postgresql',
            'database_url': database_url
        }
    
    return primary_storage, primary_config, fallback_config

# Initialize database
db = None
try:
    primary_storage, primary_config, fallback_config = get_database_config()
    
    if primary_storage and primary_config:
        db = EnhancedCloudFaceDatabase(
            storage_type=primary_storage,
            fallback_storage=fallback_config,
            **primary_config
        )
        logger.info(f"Enhanced face database initialized with {primary_storage} storage")
        logger.info(f"Database stats: {db.get_stats()}")
        logger.info(f"Sync status: {db.get_sync_status()}")
    else:
        logger.warning("No valid storage configuration found. Using empty database.")
        
except Exception as e:
    logger.error(f"Error initializing face database: {e}")
    db = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    """Convert image to Base64 encoding"""
    try:
        with Image.open(image_path) as img:
            # Limit image size to reduce memory usage
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            # Convert to RGB mode
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_data = buffer.getvalue()
            return base64.b64encode(img_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return None

def create_temp_image_path(filename):
    """Create temporary image path"""
    return os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

def get_face_image_base64(label):
    """Get face image as base64 from metadata"""
    try:
        if db is None:
            return None
        
        # Find the first face with matching label
        for i, face_label in enumerate(db.known_labels):
            if face_label == label and i < len(db.face_metadata):
                metadata = db.face_metadata[i]
                return metadata.get('image_data')
        return None
    except Exception as e:
        logger.error(f"Error getting face image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    stats = db.get_stats() if db else {'error': 'Database not initialized'}
    sync_status = db.get_sync_status() if db else {'error': 'Database not initialized'}
    
    return jsonify({
        'status': 'healthy',
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'db_initialized': db is not None,
        'storage_type': db.storage_type if db else None,
        'fallback_configured': bool(db.fallback_storage) if db else False,
        'environment_variables': {
            'github_token_set': bool(os.environ.get('GITHUB_TOKEN')),
            'gist_id_set': bool(os.environ.get('GIST_ID')),
            'database_url_set': bool(os.environ.get('DATABASE_URL')),
            's3_bucket_set': bool(os.environ.get('S3_BUCKET_NAME')),
            'mongodb_uri_set': bool(os.environ.get('MONGODB_URI'))
        },
        'db_stats': stats,
        'sync_status': sync_status
    })

@app.route('/stats')
def get_stats():
    """Get database statistics"""
    if db is None:
        return jsonify({'error': 'データベースが初期化されていません'}), 500
    
    try:
        stats = db.get_stats()
        sync_status = db.get_sync_status()
        
        return jsonify({
            'stats': stats,
            'sync_status': sync_status
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': f'統計情報の取得に失敗しました: {str(e)}'}), 500

@app.route('/sync', methods=['POST'])
def manual_sync():
    """Manual synchronization endpoint"""
    if db is None:
        return jsonify({'error': 'データベースが初期化されていません'}), 500
    
    try:
        force_sync = request.json.get('force', False) if request.json else False
        db.sync_with_cloud(force=force_sync)
        
        return jsonify({
            'success': True,
            'message': 'クラウドと同期しました',
            'sync_status': db.get_sync_status()
        })
    except Exception as e:
        logger.error(f"Manual sync error: {e}")
        return jsonify({'error': f'同期エラー: {str(e)}'}), 500

# PWA manifest and service worker
@app.route('/manifest.json')
def manifest():
    return jsonify({
        "name": "Enhanced Face Recognition App",
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
        
        # File size check
        file_data = file.read()
        if len(file_data) > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({'error': 'ファイルサイズが大きすぎます（16MB以下）'}), 400
        
        # Reset file pointer
        file.seek(0)
        
        # Generate secure filename
        filename = secure_filename(file.filename)
        filepath = create_temp_image_path(filename)
        
        # Save file
        file.save(filepath)
        
        # Check if file was saved successfully
        if not os.path.exists(filepath):
            return jsonify({'error': 'ファイルの保存に失敗しました'}), 500
        
        # Convert image to Base64
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
        logger.error(f"Upload error: {e}")
        return jsonify({'error': f'アップロードエラー: {str(e)}'}), 500

@app.route('/learn', methods=['POST'])
def learn_face():
    try:
        if db is None:
            return jsonify({'error': 'データベースが初期化されていません。ストレージ設定を確認してください。'}), 500
        
        data = request.json
        filename = data.get('filename')
        label = data.get('label')
        
        if not filename or not label:
            return jsonify({'error': 'ファイル名とラベルが必要です'}), 400
        
        filepath = create_temp_image_path(filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'ファイルが見つかりません'}), 404
        
        # Prepare metadata
        metadata = {
            'added_at': datetime.now().isoformat(),
            'source': 'web_upload',
            'image_data': image_to_base64(filepath)
        }
        
        # Add additional metadata if provided
        if 'metadata' in data:
            metadata.update(data['metadata'])
        
        # Add face to database
        success = db.add_face(filepath, label, metadata)
        
        if success:
            return jsonify({
                'success': True, 
                'message': f'{label} を学習しました',
                'stats': db.get_stats(),
                'sync_status': db.get_sync_status()
            })
        else:
            return jsonify({'error': 'データベースへの保存に失敗しました'}), 500
        
    except Exception as e:
        logger.error(f"Learn face error: {e}")
        return jsonify({'error': f'学習エラー: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict_face():
    try:
        if db is None:
            return jsonify({'error': 'データベースが初期化されていません。ストレージ設定を確認してください。'}), 500
        
        data = request.json
        filename = data.get('filename')
        threshold = data.get('threshold', 0.6)
        
        if not filename:
            return jsonify({'error': 'ファイル名が必要です'}), 400
        
        filepath = create_temp_image_path(filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'ファイルが見つかりません'}), 404
        
        # Predict face
        label, similarity, distance, metadata = db.predict(filepath, threshold)
        
        if label is None:
            return jsonify({'error': '顔が検出されませんでした'}), 400
        
        # Get matched face image
        match_image_data = None
        if label != "unknown" and metadata:
            match_image_data = metadata.get('image_data')
        
        return jsonify({
            'success': True,
            'label': label,
            'similarity': similarity,
            'distance': distance,
            'similarity_percent': f"{similarity * 100:.2f}%",
            'match_image': match_image_data,
            'metadata': metadata,
            'is_known': label != "unknown"
        })
        
    except Exception as e:
        logger.error(f"Predict face error: {e}")
        return jsonify({'error': f'予測エラー: {str(e)}'}), 500

@app.route('/confirm', methods=['POST'])
def confirm_prediction():
    try:
        if db is None:
            return jsonify({'error': 'データベースが初期化されていません。ストレージ設定を確認してください。'}), 500
        
        data = request.json
        filename = data.get('filename')
        label = data.get('label')
        is_correct = data.get('is_correct', True)
        
        if not filename:
            return jsonify({'error': 'ファイル名が必要です'}), 400
        
        filepath = create_temp_image_path(filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'ファイルが見つかりません'}), 404
        
        # Prepare metadata
        metadata = {
            'added_at': datetime.now().isoformat(),
            'source': 'confirmation',
            'original_prediction': label,
            'user_confirmed': is_correct,
            'image_data': image_to_base64(filepath)
        }
        
        if is_correct:
            # Add to existing label
            success = db.add_face(filepath, label, metadata)
            if success:
                return jsonify({
                    'success': True, 
                    'message': f'{label} に追加しました',
                    'stats': db.get_stats(),
                    'sync_status': db.get_sync_status()
                })
            else:
                return jsonify({'error': 'データベースへの保存に失敗しました'}), 500
        else:
            # Use new label
            new_label = data.get('new_label')
            if not new_label:
                return jsonify({'error': '新しいラベルが必要です'}), 400
            
            metadata['corrected_label'] = new_label
            success = db.add_face(filepath, new_label, metadata)
            if success:
                return jsonify({
                    'success': True, 
                    'message': f'{new_label} に追加しました',
                    'stats': db.get_stats(),
                    'sync_status': db.get_sync_status()
                })
            else:
                return jsonify({'error': 'データベースへの保存に失敗しました'}), 500
            
    except Exception as e:
        logger.error(f"Confirm prediction error: {e}")
        return jsonify({'error': f'確認エラー: {str(e)}'}), 500

@app.route('/labels')
def get_labels():
    """Get list of known labels"""
    if db is None:
        return jsonify({'error': 'データベースが初期化されていません'}), 500
    
    try:
        # Get unique labels
        labels = list(set(db.known_labels))
        label_info = []
        
        for label in labels:
            count = db.known_labels.count(label)
            # Get sample image for this label
            sample_image = get_face_image_base64(label)
            
            label_info.append({
                'label': label,
                'count': count,
                'sample_image': sample_image
            })
        
        return jsonify({
            'labels': labels,
            'label_info': label_info,
            'total_faces': len(db.known_faces)
        })
    except Exception as e:
        logger.error(f"Error getting labels: {e}")
        return jsonify({'error': f'ラベル取得エラー: {str(e)}'}), 500

@app.route('/backup', methods=['POST'])
def create_backup():
    """Create manual backup"""
    if db is None:
        return jsonify({'error': 'データベースが初期化されていません'}), 500
    
    try:
        db.save_database(create_backup=True)
        return jsonify({
            'success': True,
            'message': 'バックアップを作成しました',
            'sync_status': db.get_sync_status()
        })
    except Exception as e:
        logger.error(f"Backup error: {e}")
        return jsonify({'error': f'バックアップエラー: {str(e)}'}), 500

@app.route('/storage-info')
def get_storage_info():
    """Get storage configuration information"""
    if db is None:
        return jsonify({'error': 'データベースが初期化されていません'}), 500
    
    try:
        return jsonify({
            'storage_type': db.storage_type,
            'fallback_configured': bool(db.fallback_storage),
            'sync_status': db.get_sync_status(),
            'stats': db.get_stats()
        })
    except Exception as e:
        logger.error(f"Storage info error: {e}")
        return jsonify({'error': f'ストレージ情報取得エラー: {str(e)}'}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'ファイルサイズが大きすぎます'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'サーバー内部エラーが発生しました'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'リソースが見つかりません'}), 404

# Cleanup function
def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        logger.info("Temporary files cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")

if __name__ == '__main__':
    try:
        # Production environment automatically uses gunicorn
        app.run(debug=False, host='0.0.0.0', port=PORT)
    finally:
        cleanup_temp_files()