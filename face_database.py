class FaceDatabase:
    def __init__(self):
        """顔データベースの初期化"""
        self.faces = {}  # 顔データを格納する辞書
        
    def add_face(self, name, face_data):
        """顔データを追加"""
        self.faces[name] = face_data
        
    def get_face(self, name):
        """顔データを取得"""
        return self.faces.get(name)
        
    def remove_face(self, name):
        """顔データを削除"""
        if name in self.faces:
            del self.faces[name]
            return True
        return False
        
    def list_faces(self):
        """登録されている顔の一覧を取得"""
        return list(self.faces.keys())