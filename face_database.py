import face_recognition
import os
import numpy as np
from PIL import Image
import tempfile
import json
import base64
from io import BytesIO

class FaceDatabase:
    def __init__(self):
        self.known_faces = []
        self.known_labels = []
        
        # 一時ディレクトリを使用（Railway対応）
        self.temp_dir = tempfile.mkdtemp()
        self.known_faces_dir = os.path.join(self.temp_dir, "known_faces")
        
        # データベースファイルのパス（メモリ内で管理）
        self.db_file = os.path.join(self.temp_dir, "face_database.json")
        
        # known_facesディレクトリの作成
        try:
            os.makedirs(self.known_faces_dir, exist_ok=True)
            print(f"Known faces directory created: {self.known_faces_dir}")
        except Exception as e:
            print(f"Error creating known faces directory: {e}")
        
        # 既存の顔データを読み込み（本番環境では初期化時は空）
        self.load_known_faces()
    
    def load_known_faces(self):
        """既存の顔データを読み込む"""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'r') as f:
                    data = json.load(f)
                    
                # エンコーディングデータを復元
                for item in data:
                    encoding = np.array(item['encoding'])
                    label = item['label']
                    
                    self.known_faces.append(encoding)
                    self.known_labels.append(label)
                    
                print(f"Loaded {len(self.known_faces)} known faces from database")
        except Exception as e:
            print(f"Error loading known faces: {e}")
    
    def save_database(self):
        """データベースをファイルに保存"""
        try:
            data = []
            for i, encoding in enumerate(self.known_faces):
                data.append({
                    'label': self.known_labels[i],
                    'encoding': encoding.tolist()
                })
            
            with open(self.db_file, 'w') as f:
                json.dump(data, f)
                
            print(f"Database saved with {len(data)} faces")
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def add_face(self, image_path, label):
        """新しい顔を学習データに追加"""
        try:
            # 顔エンコーディングを取得
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                raise ValueError("顔が検出されませんでした")
            
            # ラベルディレクトリの作成
            label_dir = os.path.join(self.known_faces_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            
            # 画像を保存（一時的）
            image_name = f"{label}_{len([l for l in self.known_labels if l == label])}.jpg"
            save_path = os.path.join(label_dir, image_name)
            
            # PILで画像を開いて保存
            try:
                pil_image = Image.open(image_path)
                # 画像サイズを制限してメモリ使用量を減らす
                pil_image.thumbnail((400, 400), Image.Resampling.LANCZOS)
                
                # RGBモードに変換
                if pil_image.mode in ('RGBA', 'P'):
                    pil_image = pil_image.convert('RGB')
                
                pil_image.save(save_path, 'JPEG', quality=85)
            except Exception as e:
                print(f"Error saving image: {e}")
                # 画像保存に失敗してもエンコーディングは保存
            
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