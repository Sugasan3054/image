import face_recognition
import os
import numpy as np
from PIL import Image
import json
import base64
from io import BytesIO
import sqlite3
import pickle

class FaceDatabase:
    def __init__(self, use_sqlite=True, db_path="face_database.db"):
        self.known_faces = []
        self.known_labels = []
        self.use_sqlite = use_sqlite
        self.db_path = db_path
        
        if use_sqlite:
            # SQLiteデータベースを使用
            self.init_sqlite_db()
        else:
            # JSONファイルを使用（従来の方法）
            self.json_db_file = "face_database.json"
        
        # 既存の顔データを読み込み
        self.load_known_faces()
    
    def init_sqlite_db(self):
        """SQLiteデータベースを初期化"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # テーブルが存在しない場合は作成
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    encoding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # インデックスを作成
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_label ON faces(label)
            ''')
            
            conn.commit()
            conn.close()
            
            print(f"SQLite database initialized: {self.db_path}")
        except Exception as e:
            print(f"Error initializing SQLite database: {e}")
    
    def load_known_faces(self):
        """既存の顔データを読み込む"""
        try:
            if self.use_sqlite:
                self.load_from_sqlite()
            else:
                self.load_from_json()
                
            print(f"Loaded {len(self.known_faces)} known faces from database")
        except Exception as e:
            print(f"Error loading known faces: {e}")
    
    def load_from_sqlite(self):
        """SQLiteから顔データを読み込む"""
        try:
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT label, encoding FROM faces")
                rows = cursor.fetchall()
                
                for label, encoding_blob in rows:
                    # バイナリデータから配列を復元
                    encoding = pickle.loads(encoding_blob)
                    self.known_faces.append(encoding)
                    self.known_labels.append(label)
                
                conn.close()
        except Exception as e:
            print(f"Error loading from SQLite: {e}")
    
    def load_from_json(self):
        """JSONファイルから顔データを読み込む"""
        try:
            if os.path.exists(self.json_db_file):
                with open(self.json_db_file, 'r') as f:
                    data = json.load(f)
                    
                for item in data:
                    encoding = np.array(item['encoding'])
                    label = item['label']
                    
                    self.known_faces.append(encoding)
                    self.known_labels.append(label)
        except Exception as e:
            print(f"Error loading from JSON: {e}")
    
    def save_database(self):
        """データベースを保存"""
        try:
            if self.use_sqlite:
                self.save_to_sqlite()
            else:
                self.save_to_json()
                
            print(f"Database saved with {len(self.known_faces)} faces")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def save_to_sqlite(self):
        """SQLiteに保存"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 最新の顔データのみを保存（重複を避けるため、一旦クリア）
            cursor.execute("DELETE FROM faces")
            
            # 現在のデータを保存
            for i, encoding in enumerate(self.known_faces):
                encoding_blob = pickle.dumps(encoding)
                cursor.execute(
                    "INSERT INTO faces (label, encoding) VALUES (?, ?)",
                    (self.known_labels[i], encoding_blob)
                )
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving to SQLite: {e}")
    
    def save_to_json(self):
        """JSONファイルに保存"""
        try:
            data = []
            for i, encoding in enumerate(self.known_faces):
                data.append({
                    'label': self.known_labels[i],
                    'encoding': encoding.tolist()
                })
            
            with open(self.json_db_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving to JSON: {e}")
    
    def add_face(self, image_path, label):
        """新しい顔を学習データに追加"""
        try:
            # 顔エンコーディングを取得
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                raise ValueError("顔が検出されませんでした")
            
            # メモリ上のデータに追加
            self.known_faces.append(encodings[0])
            self.known_labels.append(label)
            
            # データベースを保存
            self.save_database()
            
            print(f"Added face for label: {label}")
            return True
            
        except Exception as e:
            raise Exception(f"顔の追加に失敗しました: {str(e)}")
    
    def predict(self, image_path):
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
            
            # 類似度を計算（距離が小さいほど類似度が高い）
            similarity = 1 - min_distance
            
            # 閾値を設定（0.6以上で一致とみなす）
            threshold = 0.6
            
            if similarity >= threshold:
                predicted_label = self.known_labels[min_distance_index]
                return predicted_label, similarity, min_distance
            else:
                return "unknown", similarity, min_distance
                
        except Exception as e:
            raise Exception(f"顔の予測に失敗しました: {str(e)}")
    
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
            'labels': dict([(label, self.get_face_count(label)) for label in set(self.known_labels)])
        }
    
    def backup_database(self, backup_path):
        """データベースをバックアップ"""
        try:
            if self.use_sqlite:
                # SQLiteファイルをコピー
                import shutil
                shutil.copy2(self.db_path, backup_path)
            else:
                # JSONファイルをコピー
                import shutil
                shutil.copy2(self.json_db_file, backup_path)
            
            print(f"Database backed up to: {backup_path}")
        except Exception as e:
            print(f"Error backing up database: {e}")
    
    def restore_database(self, backup_path):
        """バックアップからデータベースを復元"""
        try:
            if self.use_sqlite:
                import shutil
                shutil.copy2(backup_path, self.db_path)
            else:
                import shutil
                shutil.copy2(backup_path, self.json_db_file)
            
            # データを再読み込み
            self.known_faces = []
            self.known_labels = []
            self.load_known_faces()
            
            print(f"Database restored from: {backup_path}")
        except Exception as e:
            print(f"Error restoring database: {e}")

# 使用例
if __name__ == "__main__":
    # SQLiteを使用する場合
    db = FaceDatabase(use_sqlite=True, db_path="persistent_face_db.db")
    
    # JSONを使用する場合
    # db = FaceDatabase(use_sqlite=False)
    
    # 統計情報を表示
    print("Database stats:", db.get_stats())