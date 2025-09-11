# -----------------------------
# InsightFaceの初期化
# -----------------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))  # GPU利用 (CPUならctx_id=-1)

known_face_encodings = []
known_face_names = []
