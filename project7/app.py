from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import os
import threading
import time
from tqdm import tqdm
import base64
import json
from tqdm import tqdm
from numpy.linalg import norm
from insightface.app import FaceAnalysis

app = Flask(__name__)
CORS(app)

# グローバル変数
video_capture = None
frame_data = None
face_data = []
is_processing = False
tolerance = 0.4
known_face_encodings = []
known_face_names = []

# 年齢推定モデルの読み込み
AGE_PROTO = "./model/deploy_age.prototxt"
AGE_MODEL = "./model/age_net.caffemodel"
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60+)']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def load_known_faces():
    """既知の顔データを読み込む"""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    image_folder = "images"
    if not os.path.exists(image_folder):
        return
    
    folder_list = [folder for folder in os.listdir(image_folder) 
                   if os.path.isdir(os.path.join(image_folder, folder))]
    
    for foldername in folder_list:
        folder_path = os.path.join(image_folder, foldername)
        image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
        
        for filename in image_files:
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(foldername)

def process_frame(frame):
    """フレームを処理して顔認識と年齢推定を行う"""
    global face_data
    
    # フレームをより小さくして処理速度を向上
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)  # 0.25から0.2に変更
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    
    # 顔検出（より高速なモデルを使用）
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")  # CNNからHOGに変更
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    # 座標を元のサイズに戻す（5倍に変更）
    face_locations = [(top * 5, right * 5, bottom * 5, left * 5)
                      for (top, right, bottom, left) in face_locations]
    
    current_faces = []
    
    # 最大3つの顔のみ処理（パフォーマンス向上）
    for i, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings)):
        if i >= 3:  # 最大3つの顔のみ処理
            break
            
        # 年齢推定（簡略化）
        face_img = frame[top:bottom, left:right].copy()
        age = "Unknown"
        if face_img.size > 0 and face_img.shape[0] > 50 and face_img.shape[1] > 50:  # 最小サイズチェック
            try:
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = AGE_LIST[age_preds[0].argmax()]
            except:
                age = "Unknown"
        
        # 顔認識（簡略化）
        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            top_matches_idx = np.argsort(face_distances)[:2]  # 上位2つのみ
            
            top_names = []
            for idx in top_matches_idx:
                dist = face_distances[idx]
                if dist <= tolerance:
                    similarity = max(0, (1 - dist / tolerance) * 100)
                    top_names.append({
                        "name": known_face_names[idx],
                        "similarity": round(similarity, 1)
                    })
                else:
                    top_names.append({
                        "name": "Unknown",
                        "similarity": 0.0
                    })
        else:
            top_names = [{"name": "Unknown", "similarity": 0.0}]
        
        current_faces.append({
            "bbox": [int(left), int(top), int(right), int(bottom)],
            "age": age,
            "matches": top_names
        })
    
    face_data = current_faces

def video_processing():
    """ビデオ処理のメインループ"""
    global video_capture, frame_data, is_processing
    
    video_capture = cv2.VideoCapture(0)
    # カメラの解像度を下げてパフォーマンス向上
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    skip_frames = 10  # より多くのフレームをスキップ
    
    while is_processing:
        ret, frame = video_capture.read()
        if not ret:
            continue
        
        # フレーム処理（スキップしてパフォーマンス向上）
        if frame_count % skip_frames == 0:
            process_frame(frame)
        
        frame_count += 1
        
        # フレームをエンコード（品質を下げてパフォーマンス向上）
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        time.sleep(0.016)  # 60FPS制限（より滑らか）
    
    video_capture.release()

@app.route('/')
def index():
    """メインページ"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """ビデオフィードのエンドポイント"""
    def generate():
        global frame_data
        while is_processing:
            if frame_data:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       base64.b64decode(frame_data) + b'\r\n')
            time.sleep(0.03)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_data')
def get_face_data():
    """顔認識データのエンドポイント"""
    return jsonify(face_data)

@app.route('/settings', methods=['POST'])
def update_settings():
    """設定の更新"""
    global tolerance
    data = request.get_json()
    if 'tolerance' in data:
        tolerance = float(data['tolerance'])
    return jsonify({"status": "success"})

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """カメラの開始"""
    global is_processing
    if not is_processing:
        is_processing = True
        threading.Thread(target=video_processing, daemon=True).start()
    return jsonify({"status": "started"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """カメラの停止"""
    global is_processing
    is_processing = False
    return jsonify({"status": "stopped"})

if __name__ == '__main__':
    # 既知の顔データを読み込み
    load_known_faces()
    print("既知の顔データの読み込み完了")
    
    app.run(debug=True, host='127.0.0.1', port=8080)
