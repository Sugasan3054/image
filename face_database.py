import face_recognition
import os
import numpy as np
from PIL import Image
import json
import base64
from io import BytesIO
import pickle
import requests
from urllib.parse import urlparse

class CloudFaceDatabase:
    def __init__(self, storage_type="github", **kwargs):
        """
        storage_type: "github", "s3", "gcs", "postgresql", "mongodb"
        """
        self.known_faces = []
        self.known_labels = []
        self.storage_type = storage_type
        self.config = kwargs
        
        # ストレージタイプに応じた初期化
        if storage_type == "github":
            self.init_github_storage()
        elif storage_type == "postgresql":
            self.init_postgresql()
        elif storage_type == "s3":
            self.init_s3_storage()
        elif storage_type == "mongodb":
            self.init_mongodb()
        
        # 既存データの読み込み
        self.load_known_faces()
    
    def init_github_storage(self):
        """GitHub Gist をストレージとして使用"""
        self.github_token = self.config.get('github_token')
        self.gist_id = self.config.get('gist_id')
        self.gist_filename = self.config.get('gist_filename', 'face_database.json')
        
        if not self.github_token:
            raise ValueError("GitHub token is required for GitHub storage")
    
    def init_postgresql(self):
        """PostgreSQL データベースを初期化"""
        try:
            import psycopg2
            
            self.db_url = self.config.get('database_url') or os.environ.get('DATABASE_URL')
            if not self.db_url:
                raise ValueError("Database URL is required for PostgreSQL storage")
            
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            # テーブル作成
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id SERIAL PRIMARY KEY,
                    label VARCHAR(255) NOT NULL,
                    encoding BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_faces_label ON faces(label)
            ''')
            
            conn.commit()
            conn.close()
            
        except ImportError:
            raise ImportError("psycopg2 is required for PostgreSQL storage. Install with: pip install psycopg2-binary")
    
    def init_s3_storage(self):
        """Amazon S3 ストレージを初期化"""
        try:
            import boto3
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.config.get('aws_access_key_id'),
                aws_secret_access_key=self.config.get('aws_secret_access_key'),
                region_name=self.config.get('aws_region', 'us-east-1')
            )
            self.bucket_name = self.config.get('bucket_name')
            self.s3_key = self.config.get('s3_key', 'face_database.json')
            
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
    
    def init_mongodb(self):
        """MongoDB を初期化"""
        try:
            import pymongo
            
            self.mongo_url = self.config.get('mongo_url') or os.environ.get('MONGODB_URI')
            if not self.mongo_url:
                raise ValueError("MongoDB URL is required for MongoDB storage")
            
            self.client = pymongo.MongoClient(self.mongo_url)
            self.db = self.client.face_recognition
            self.collection = self.db.faces
            
        except ImportError:
            raise ImportError("pymongo is required for MongoDB storage. Install with: pip install pymongo")
    
    def load_known_faces(self):
        """ストレージから顔データを読み込む"""
        try:
            if self.storage_type == "github":
                self.load_from_github()
            elif self.storage_type == "postgresql":
                self.load_from_postgresql()
            elif self.storage_type == "s3":
                self.load_from_s3()
            elif self.storage_type == "mongodb":
                self.load_from_mongodb()
            
            print(f"Loaded {len(self.known_faces)} known faces from {self.storage_type}")
        except Exception as e:
            print(f"Error loading known faces: {e}")
    
    def load_from_github(self):
        """GitHub Gist から読み込み"""
        try:
            if not self.gist_id:
                print("No gist_id provided, starting with empty database")
                return
            
            url = f"https://api.github.com/gists/{self.gist_id}"
            headers = {"Authorization": f"token {self.github_token}"}
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                gist_data = response.json()
                content = gist_data['files'][self.gist_filename]['content']
                data = json.loads(content)
                
                for item in data:
                    encoding = np.array(item['encoding'])
                    label = item['label']
                    
                    self.known_faces.append(encoding)
                    self.known_labels.append(label)
            else:
                print(f"Failed to load from GitHub: {response.status_code}")
        except Exception as e:
            print(f"Error loading from GitHub: {e}")
    
    def load_from_postgresql(self):
        """PostgreSQL から読み込み"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            cursor.execute("SELECT label, encoding FROM faces")
            rows = cursor.fetchall()
            
            for label, encoding_bytes in rows:
                encoding = pickle.loads(encoding_bytes)
                self.known_faces.append(encoding)
                self.known_labels.append(label)
            
            conn.close()
        except Exception as e:
            print(f"Error loading from PostgreSQL: {e}")
    
    def load_from_s3(self):
        """Amazon S3 から読み込み"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.s3_key)
            content = response['Body'].read().decode('utf-8')
            data = json.loads(content)
            
            for item in data:
                encoding = np.array(item['encoding'])
                label = item['label']
                
                self.known_faces.append(encoding)
                self.known_labels.append(label)
        except Exception as e:
            print(f"Error loading from S3: {e}")
    
    def load_from_mongodb(self):
        """MongoDB から読み込み"""
        try:
            for doc in self.collection.find():
                encoding = np.array(doc['encoding'])
                label = doc['label']
                
                self.known_faces.append(encoding)
                self.known_labels.append(label)
        except Exception as e:
            print(f"Error loading from MongoDB: {e}")
    
    def save_database(self):
        """データベースを保存"""
        try:
            if self.storage_type == "github":
                self.save_to_github()
            elif self.storage_type == "postgresql":
                self.save_to_postgresql()
            elif self.storage_type == "s3":
                self.save_to_s3()
            elif self.storage_type == "mongodb":
                self.save_to_mongodb()
            
            print(f"Database saved to {self.storage_type} with {len(self.known_faces)} faces")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def save_to_github(self):
        """GitHub Gist に保存"""
        try:
            data = []
            for i, encoding in enumerate(self.known_faces):
                data.append({
                    'label': self.known_labels[i],
                    'encoding': encoding.tolist()
                })
            
            content = json.dumps(data, indent=2)
            
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
            
            headers = {"Authorization": f"token {self.github_token}"}
            
            if self.gist_id:
                response = requests.patch(url, json=payload, headers=headers)
            else:
                response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code in [200, 201]:
                if not self.gist_id:
                    self.gist_id = response.json()['id']
                    print(f"Created new gist: {self.gist_id}")
            else:
                print(f"Failed to save to GitHub: {response.status_code}")
        except Exception as e:
            print(f"Error saving to GitHub: {e}")
    
    def save_to_postgresql(self):
        """PostgreSQL に保存"""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            # 既存データを削除
            cursor.execute("DELETE FROM faces")
            
            # 新しいデータを挿入
            for i, encoding in enumerate(self.known_faces):
                encoding_bytes = pickle.dumps(encoding)
                cursor.execute(
                    "INSERT INTO faces (label, encoding) VALUES (%s, %s)",
                    (self.known_labels[i], encoding_bytes)
                )
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving to PostgreSQL: {e}")
    
    def save_to_s3(self):
        """Amazon S3 に保存"""
        try:
            data = []
            for i, encoding in enumerate(self.known_faces):
                data.append({
                    'label': self.known_labels[i],
                    'encoding': encoding.tolist()
                })
            
            content = json.dumps(data, indent=2)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.s3_key,
                Body=content,
                ContentType='application/json'
            )
        except Exception as e:
            print(f"Error saving to S3: {e}")
    
    def save_to_mongodb(self):
        """MongoDB に保存"""
        try:
            # 既存データを削除
            self.collection.delete_many({})
            
            # 新しいデータを挿入
            documents = []
            for i, encoding in enumerate(self.known_faces):
                documents.append({
                    'label': self.known_labels[i],
                    'encoding': encoding.tolist()
                })
            
            if documents:
                self.collection.insert_many(documents)
        except Exception as e:
            print(f"Error saving to MongoDB: {e}")
    
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
            
            # クラウドに保存
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
            
            # 類似度を計算
            similarity = 1 - min_distance
            
            # 閾値を設定
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

# 使用例
if __name__ == "__main__":
    # GitHub Gist を使用
    github_db = CloudFaceDatabase(
        storage_type="github",
        github_token="your_github_token",
        gist_id="your_gist_id"  # 初回は None
    )
    
    # PostgreSQL を使用 (Heroku、Railway等)
    # postgresql_db = CloudFaceDatabase(
    #     storage_type="postgresql",
    #     database_url="postgresql://user:pass@host:port/db"
    # )
    
    # Amazon S3 を使用
    # s3_db = CloudFaceDatabase(
    #     storage_type="s3",
    #     aws_access_key_id="your_access_key",
    #     aws_secret_access_key="your_secret_key",
    #     bucket_name="your_bucket",
    #     s3_key="face_database.json"
    # )
    
    # MongoDB を使用
    # mongo_db = CloudFaceDatabase(
    #     storage_type="mongodb",
    #     mongo_url="mongodb://user:pass@host:port/db"
    # )
    
    print("Database stats:", github_db.get_stats())