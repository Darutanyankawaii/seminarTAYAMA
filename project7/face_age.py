import cv2
import face_recognition
import numpy as np
import os
from tqdm import tqdm

tolerance = 0.4
image_folder = "images"

# 年齢推定モデルの読み込み
AGE_PROTO = "./model/deploy_age.prototxt"
AGE_MODEL = "./model/age_net.caffemodel"
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

# 年齢クラス（学習済みモデルに依存）
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60+)']

# mean値（モデルによって指定される）
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

known_face_encodings = []
known_face_names = []

# 既知データ読み込み
folder_list = [folder for folder in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, folder))]
for foldername in tqdm(folder_list, desc="Processing Folders"):
    folder_path = os.path.join(image_folder, foldername)
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]

    for filename in tqdm(image_files, desc=f"Processing Images in {foldername}", leave=False):
        image_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(foldername)

print("All complete.")

# カメラ
video_capture = cv2.VideoCapture(0)

frame_count = 0
skip_frames = 5  # 5フレームに1回だけ処理

ret, frame = video_capture.read()

small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

face_locations = face_recognition.face_locations(rgb_small_frame)
face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
face_locations = [(top * 4, right * 4, bottom * 4, left * 4)
                          for (top, right, bottom, left) in face_locations]


while True:
    ret, frame = video_capture.read()
    
    if frame_count % skip_frames == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_locations = [(top * 4, right * 4, bottom * 4, left * 4)
                          for (top, right, bottom, left) in face_locations]
    
    frame_count += 1

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # --- 年齢推定の処理をここで追加 ---
        face_img = frame[top:bottom, left:right].copy()
        if face_img.size > 0:  # 顔が切り出せたときだけ処理
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]
        else:
            age = "Unknown"

    # --- 顔認識処理 ---
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    # 距離が近い順に3人
        top_matches_idx = np.argsort(face_distances)[:3]

        top_names = []
        for idx in top_matches_idx:
            dist = face_distances[idx]
            if dist <= tolerance:
                similarity = max(0, (1 - dist / tolerance) * 100)  # 類似度%
                top_names.append(f"{known_face_names[idx]} ({similarity:.1f}%)")
            else:
                top_names.append(f"Unknown (0.0%)")

    # --- 描画処理 ---
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX

    # 複数行表示（名前候補）
        for i, name in enumerate(top_names):
            y_offset = bottom + 20 + (i * 20)
            cv2.putText(frame, name, (left + 6, y_offset), font, 0.5, (255, 255, 255), 1)

    # 年齢ラベルを表示（顔の下）
        cv2.putText(frame, f"Age: {age}", (left + 6, bottom + 100), font, 0.6, (0, 255, 0), 1)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
