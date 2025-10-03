# backend.py with fix for RGB conversion in classify_crop

from flask import Flask, request
from werkzeug.utils import secure_filename
import os
import cv2
import json
from datetime import datetime
from pathlib import Path
import threading
import queue
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import mediapipe as mp
from ultralytics import YOLO

# Constants from two.py
YOLO_MODEL_PATH = "yolov8n.pt"
CLOTH_CLASSIFIER_PATH = "clothing_classifier.pth"
OUTPUT_JSON = "frame_log_with_clothing_and_actions.json"
CONF_THRESH = 0.35
WANTED_OBJECTS = {"person", "car", "laptop", "cell phone", "book", "bottle", "handbag", "gun", "knife", "airpods", "charger"}
WANTED_PERSON_LABELS = {"person"}
CLOTHING_CLASSES = ["army", "doctor", "civilian", "other"]
ACTION_CLASSES = ["running", "crouching", "aiming_weapon", "static"]
SEQ_LENGTH = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VARIANCE_THRESHOLD = 0.01

# MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Helpers
def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else []

def save_json(p, data):
    Path(p).write_text(json.dumps(data, indent=2))

def heuristic_classify(crop_bgr):
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    h_med = int(np.median(h))
    s_med = int(np.median(s))
    if v.mean() > 200 and s.mean() < 40:
        return "doctor", 0.6
    if 40 <= h_med <= 90 and s_med > 50:
        return "army", 0.5
    return "civilian", 0.5

# LSTM for action
class ActionLSTM(nn.Module):
    def __init__(self, input_size=33*3, hidden_size=128, num_classes=4):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# Processor class (adapted from VideoProcessor)
class Processor:
    def __init__(self):
        # Load YOLO
        self.model = YOLO(YOLO_MODEL_PATH)
        self.names_map = self.model.model.names if hasattr(self.model, "model") else self.model.names

        # Load clothing classifier
        self.classifier = None
        try:
            clf = models.resnet18(weights=None)
            clf.fc = nn.Linear(clf.fc.in_features, len(CLOTHING_CLASSES))
            clf.load_state_dict(torch.load(CLOTH_CLASSIFIER_PATH, map_location=DEVICE, weights_only=True))
            clf.to(DEVICE).eval()
            self.classifier = clf
        except Exception as e:
            print(f"Using heuristic classifier. Error: {e}")

        self.clf_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load action model
        self.action_model = ActionLSTM().to(DEVICE)
        try:
            self.action_model.load_state_dict(torch.load("action_lstm.pth", map_location=DEVICE, weights_only=True))
        except:
            print("No action model found, using random init for demo.")
        self.action_model.eval()

        # MediaPipe Pose
        self.pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Logging
        self.log_records = load_json(OUTPUT_JSON)
        self.current_record = None
        self.prev_summary = None
        self.frame_idx = 0
        self.lock = threading.Lock()

        # Keypoints queue and EMA
        self.keypoints_queue = queue.deque(maxlen=SEQ_LENGTH)
        self.ema_keypoints = None
        self.ema_alpha = 0.7

    def classify_crop(self, crop):
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # Convert to RGB for classifier
        if self.classifier is None:
            return heuristic_classify(crop)
        try:
            img_t = self.clf_transforms(crop_rgb).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = self.classifier(img_t)
                probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
                idx = int(probs.argmax())
                return CLOTHING_CLASSES[idx], float(probs[idx])
        except Exception as e:
            print(f"Classifier error: {e}")
            return heuristic_classify(crop)

    def compute_keypoints_variance(self, keypoints_seq):
        if len(keypoints_seq) < 2:
            return float('inf')
        keypoints_array = np.array(keypoints_seq)
        coords = keypoints_array[:, :-1:3]
        return np.var(coords, axis=0).mean()

    def predict_action(self, keypoints_seq):
        if len(keypoints_seq) < SEQ_LENGTH:
            return "unknown", 0.0
        variance = self.compute_keypoints_variance(keypoints_seq)
        if variance < VARIANCE_THRESHOLD:
            return "static", 0.9
        seq_tensor = torch.tensor([keypoints_seq]).float().to(DEVICE)
        with torch.no_grad():
            output = self.action_model(seq_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
            idx = int(probs.argmax())
            return ACTION_CLASSES[idx], float(probs[idx])

    def process_frame(self, img):
        self.frame_idx += 1

        # YOLO Detection
        results = self.model(img, conf=CONF_THRESH, imgsz=640)
        objects_in_frame = []
        persons_data = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf)
                if conf < CONF_THRESH:
                    continue
                cls_id = int(box.cls)
                label = self.names_map.get(cls_id, str(cls_id))
                if label not in WANTED_OBJECTS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label in WANTED_PERSON_LABELS:
                    # Crop for clothing and pose
                    pad = int(0.05 * max(x2 - x1, y2 - y1))
                    x1c, y1c = max(0, x1 - pad), max(0, y1 - pad)
                    x2c, y2c = min(img.shape[1] - 1, x2 + pad), min(img.shape[0] - 1, y2 + pad)
                    crop = img[y1c:y2c, x1c:x2c].copy()
                    if crop.size == 0:
                        continue

                    # Clothing classification
                    category, score = self.classify_crop(crop)

                    # Pose
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose.process(rgb_crop)
                    if pose_results.pose_landmarks:
                        keypoints = []
                        for lm in pose_results.pose_landmarks.landmark:
                            keypoints.extend([lm.x, lm.y, lm.visibility])
                        if self.ema_keypoints is None:
                            self.ema_keypoints = np.array(keypoints)
                        else:
                            self.ema_keypoints = self.ema_alpha * np.array(keypoints) + (1 - self.ema_alpha) * self.ema_keypoints
                        self.keypoints_queue.append(self.ema_keypoints.tolist())
                        action, action_conf = self.predict_action(list(self.keypoints_queue))
                    else:
                        action, action_conf = "no_pose", 0.0

                    persons_data.append({"category": category, "action": action})

                objects_in_frame.append({"label": label, "conf": conf})

        # Logging
        summary_objects = sorted([f"{o['label']}" for o in objects_in_frame])
        summary_persons = sorted([f"{p['category']}:{p['action']}" for p in persons_data])
        summary = summary_objects + summary_persons

        with self.lock:
            if summary != self.prev_summary:
                if self.current_record:
                    self.current_record["end_time"] = now_iso()
                    self.log_records.append(self.current_record)
                    save_json(OUTPUT_JSON, self.log_records)
                
                self.current_record = {
                    "frame": self.frame_idx,
                    "start_time": now_iso(),
                    "end_time": None,
                    "objects": [o["label"] for o in objects_in_frame],
                    "persons": persons_data
                }

            self.prev_summary = summary

# Function to process video
def process_video(file_path):
    processor = Processor()
    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processor.process_frame(frame)
    cap.release()

    # Close current record
    with processor.lock:
        if processor.current_record:
            processor.current_record["end_time"] = now_iso()
            processor.log_records.append(processor.current_record)
            save_json(OUTPUT_JSON, processor.log_records)

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video file', 400
    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the video
        process_video(file_path)
        
        # Generate SITREP using the separate LLM file
        from llm_task import generate_sitrep  # Import here or at top
        sitrep = generate_sitrep(OUTPUT_JSON)
        
        # Optionally clean up
        os.remove(file_path)
        
        return sitrep

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)