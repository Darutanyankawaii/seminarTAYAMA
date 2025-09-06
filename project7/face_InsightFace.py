import cv2
import numpy as np
import os
from tqdm import tqdm
from numpy.linalg import norm
from insightface.app import FaceAnalysis

tolerance = 0.4
image_folder = "images"

# -----------------------------
# 年齢推定モデルの読み込み
# -----------------------------
AGE_PROTO = "./model/deploy_age.prototxt"
AGE_MODEL = "./model/age_net.caffemodel"
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60+)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# -----------------------------
# InsightFaceの初期化
# -----------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))  # GPU利用 (CPUならctx_id=-1)

known_face_encodings = []
known_face_names = []

# -----------------------------
# 既知顔データの準備
# -----------------------------
folder_list = [folder for folder in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, folder))]
for foldername in tqdm(folder_list, desc="Processing Folders"):
    folder_path = os.path.join(image_folder, foldername)
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

    for filename in tqdm(image_files, desc=f"Processing Images in {foldername}", leave=False):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        results = app.get(image)
        if results:
            known_face_encodings.append(results[0].embedding)
            known_face_names.append(foldername)

print("All complete.")

# -----------------------------
# 類似度計算（コサイン類似度）
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# -----------------------------
# カメラ起動
# -----------------------------
video_capture = cv2.VideoCapture(0)
frame_count = 0
skip_frames = 5

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    if frame_count % skip_frames == 0:
        faces = app.get(frame)
    frame_count += 1

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.embedding

        # --- 年齢推定 ---
        face_img = frame[y1:y2, x1:x2].copy()
        if face_img.size > 0:
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]
        else:
            age = "Unknown"

        # --- 顔認識 ---
        similarities = [cosine_similarity(embedding, known) for known in known_face_encodings]

        # 上位3件
        top_matches_idx = np.argsort(similarities)[::-1][:3]
        top_names = []
        for idx in top_matches_idx:
            sim = similarities[idx]
            if sim >= (1 - tolerance):  # 類似度しきい値
                top_names.append(f"{known_face_names[idx]} ({sim*100:.1f}%)")
            else:
                top_names.append(f"Unknown (0.0%)")

        # --- 描画処理 ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX

        # 名前候補の複数行表示
        for i, name in enumerate(top_names):
            y_offset = y2 + 20 + (i * 20)
            cv2.putText(frame, name, (x1 + 6, y_offset), font, 0.5, (255, 255, 255), 1)

        # 年齢ラベル
        cv2.putText(frame, f"Age: {age}", (x1 + 100, y2 + 50), font, 0.6, (0, 255, 0), 1)

    cv2.imshow('camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
