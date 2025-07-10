import face_recognition
import os
import numpy as np
from PIL import Image

class FaceDatabase:
    def __init__(self):
        self.known_faces = []
        self.known_labels = []
        self.known_faces_dir = "known_faces"
        
        # known_facesディレクトリの作成
        os.makedirs(self.known_faces_dir, exist_ok=True)
        
        # 既存の顔データを読み込み
        self.load_known_faces()
    
    def load_known_faces(self):
        """既存の顔データを読み込む"""
        if not os.path.exists(self.known_faces_dir):
            return
        
        for label in os.listdir(self.known_faces_dir):
            label_dir = os.path.join(self.known_faces_dir, label)
            if os.path.isdir(label_dir):
                for image_file in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_file)
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            # 顔エンコーディングを取得
                            image = face_recognition.load_image_file(image_path)
                            encodings = face_recognition.face_encodings(image)
                            
                            if encodings:
                                self.known_faces.append(encodings[0])
                                self.known_labels.append(label)
                        except Exception as e:
                            print(f"顔の読み込みに失敗: {image_path}, エラー: {e}")
    
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
            
            # 画像を保存
            image_name = f"{label}_{len(os.listdir(label_dir))}.jpg"
            save_path = os.path.join(label_dir, image_name)
            
            # PILで画像を開いて保存
            pil_image = Image.open(image_path)
            pil_image.save(save_path)
            
            # メモリ上のデータに追加
            self.known_faces.append(encodings[0])
            self.known_labels.append(label)
            
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