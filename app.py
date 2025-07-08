import tkinter as tk 
from tkinter import filedialog, messagebox, simpledialog 
from PIL import Image, ImageTk, ImageOps 
import os 
from face_database import FaceDatabase 
 
class FaceApp: 
    def __init__(self, root): 
        self.root = root 
        self.root.title("老若認証") 
        self.root.configure(bg="black") 
        self.root.geometry("600x400") 
        self.root.resizable(False, False) 
 
        self.db = FaceDatabase() 
        self.image_path = None 
        self.predicted_label = None 
 
        self.build_ui() 
 
    def build_ui(self): 
        style = { 
            'bg': 'black', 'fg': 'white', 'font': ('Arial', 11) 
        } 
 
        btn_frame = tk.Frame(self.root, bg='black') 
        btn_frame.pack(fill=tk.X, pady=5) 
 
        for text, cmd in [ 
            ("画像を選択", self.select_image), 
            ("学習", self.learn_face), 
            ("推定", self.predict_face) 
        ]: 
            tk.Button(btn_frame, text=text, command=cmd, bg="#333", fg="white", font=('Arial', 11), relief=tk.FLAT).pack(side=tk.LEFT, padx=5) 
 
        self.status = tk.StringVar() 
        self.status.set("準備完了") 
        tk.Label(self.root, textvariable=self.status, **style).pack() 
 
        self.canvas_frame = tk.Frame(self.root, bg='black') 
        self.canvas_frame.pack() 
 
        self.selected_canvas = tk.Label(self.canvas_frame, bg='black') 
        self.selected_canvas.pack(side=tk.LEFT, padx=10) 
 
        self.match_canvas = tk.Label(self.canvas_frame, bg='black') 
        self.match_canvas.pack(side=tk.LEFT, padx=10) 
 
        self.similarity_label = tk.Label(self.root, text="", **style) 
        self.similarity_label.pack(pady=5) 
 
        self.confirm_frame = tk.Frame(self.root, bg='black') 
        self.confirm_frame.pack(pady=5) 
 
        self.confirm_label = tk.Label(self.confirm_frame, text="この人物は一致していますか？", **style) 
        self.confirm_label.pack(side=tk.LEFT, padx=10) 
 
        self.yes_button = tk.Button(self.confirm_frame, text="はい", command=self.confirm_yes, bg="#006400", fg="white", font=('Arial', 10), relief=tk.FLAT) 
        self.no_button = tk.Button(self.confirm_frame, text="いいえ", command=self.confirm_no, bg="#8B0000", fg="white", font=('Arial', 10), relief=tk.FLAT) 
        self.yes_button.pack(side=tk.LEFT, padx=5) 
        self.no_button.pack(side=tk.LEFT, padx=5) 
 
        self.confirm_frame.pack_forget() 
 
    def select_image(self): 
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")]) 
        if path: 
            self.image_path = path 
            self.show_image(self.selected_canvas, path) 
            self.status.set(f"選択中: {os.path.basename(path)}") 
            self.similarity_label.config(text="") 
            self.match_canvas.config(image="") 
            self.confirm_frame.pack_forget() 
 
    def show_image(self, widget, path): 
        image = Image.open(path) 
        image = ImageOps.fit(image, (250, 250), method=Image.Resampling.LANCZOS) 
        photo = ImageTk.PhotoImage(image) 
        widget.configure(image=photo) 
        widget.image = photo 
 
    def learn_face(self): 
        if not self.image_path: 
            messagebox.showwarning("警告", "画像を選択してください") 
            return 
 
        label = simpledialog.askstring("ラベル入力", "この人物のラベル名を入力してください:") 
        if not label: 
            return 
 
        try: 
            self.db.add_face(self.image_path, label) 
            messagebox.showinfo("学習完了", f"{label} を学習しました") 
            self.status.set(f"{label} を学習済み") 
        except Exception as e: 
            messagebox.showerror("エラー", str(e)) 
 
    def predict_face(self): 
        if not self.image_path: 
            messagebox.showwarning("警告", "画像を選択してください") 
            return 
 
        label, similarity, _ = self.db.predict(self.image_path) 
        if label is None: 
            messagebox.showinfo("結果", "顔が検出されませんでした") 
            return 
 
        label_dir = os.path.join("known_faces", label) 
        image_list = os.listdir(label_dir) 
        if image_list: 
            match_path = os.path.join(label_dir, image_list[0]) 
            self.show_image(self.match_canvas, match_path) 
 
        similarity_pct = f"類似度: {similarity * 100:.2f}%  (ラベル: {label})" 
        self.similarity_label.config(text=similarity_pct) 
        self.predicted_label = label 
        self.confirm_frame.pack() 
 
    def confirm_yes(self): 
        if self.image_path and self.predicted_label: 
            self.db.add_face(self.image_path, self.predicted_label) 
            self.status.set(f"{self.predicted_label} に追加しました") 
            self.confirm_frame.pack_forget() 
 
    def confirm_no(self): 
        if not self.image_path: 
            return 
        new_label = simpledialog.askstring("ラベル修正", "正しいラベル名を入力してください:") 
        if new_label: 
            self.db.add_face(self.image_path, new_label) 
            self.status.set(f"{new_label} に追加しました") 
            self.confirm_frame.pack_forget() 
 
if __name__ == "__main__": 
    root = tk.Tk() 
    app = FaceApp(root) 
    root.mainloop() 