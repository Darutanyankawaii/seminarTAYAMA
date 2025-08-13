import cv2
import face_recognition
import numpy as np
import os
from tqdm import tqdm  # プログレスバー表示ライブラリ

tolerance = 0.4

# 画像フォルダのパス
image_folder = "./images"

# 画像を読み込み、顔の特徴を記録する
known_face_encodings = []
known_face_names = []

# フォルダごとに画像を処理
folder_list = [folder for folder in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, folder))]

# tqdm を使ってフォルダの処理状況を表示
for foldername in tqdm(folder_list, desc="Processing Folders"):
    folder_path = os.path.join(image_folder, foldername)
    
    # フォルダ内の画像ファイルを処理
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    
    # tqdm を使ってフォルダ内の画像ファイルの処理状況を表示
    for filename in tqdm(image_files, desc=f"Processing Images in {foldername}", leave=False):
        image_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        
        if encoding:
            # 複数の画像から同じ人物の特徴を追加
            known_face_encodings.append(encoding[0])
            known_face_names.append(foldername)  # フォルダ名を表示名として取得

print("All complete.")

# カメラを初期化
video_capture = cv2.VideoCapture(0)

process_this_frame = True

while True:
    ret, frame = video_capture.read()

    if process_this_frame:
        # ここで顔認識処理
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        
        # 顔検出と顔認識
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # 顔の座標を拡大して元のフレームに戻す
        face_locations = [(top * 4, right * 4, bottom * 4, left * 4) for (top, right, bottom, left) in face_locations]

    # 毎フレーム処理を行わないようにする
    process_this_frame = not process_this_frame
    
    # 各顔に対して認識
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance) # 厳しくするならtoleranceを小さく
        name = "Unknown"
        
        # マッチした顔があれば名前を取得
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        # 顔の周りにボックスを描画
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    
    # フレームを表示
    cv2.imshow('Video', frame)
    
    # 'q'キーが押されたらループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラを解放し、ウィンドウを閉じる
video_capture.release()
cv2.destroyAllWindows()

