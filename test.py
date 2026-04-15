import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque


SIGNS      = ['yes', 'no', 'question', 'repeat', 'ok', 'wait']
FRAMES     = 30
THRESHOLD  = 0.85
MODEL_PATH = 'isl_model.keras'
TASK_PATH  = r'C:\Users\samth\Downloads\Indian-sign-language-using-OpenCV-main\Indian-sign-language-using-OpenCV-main\hand_landmarker.task'


model = load_model(MODEL_PATH)

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

def extract_landmarks(result):
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand]).flatten()
    return np.zeros(63)

def draw_hands(frame, result):
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for lm in hand:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)


sequence   = deque(maxlen=FRAMES)
prediction = ''
confidence = 0.0
pred_color = (0, 255, 0)

# ✅ FIXED: changed 1 → 0 for default webcam
cap = cv2.VideoCapture(0)

# ✅ NEW: fail loudly if camera not found
if not cap.isOpened():
    print("ERROR: Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1)")
    exit()

print("Webcam opened successfully! Press 'q' to quit.")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame from webcam.")
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)

        draw_hands(frame, result)

        landmarks = extract_landmarks(result)
        sequence.append(landmarks)

        if len(sequence) == FRAMES:
            input_data = np.expand_dims(list(sequence), axis=0)
            predictions = model.predict(input_data, verbose=0)[0]
            confidence  = np.max(predictions)
            pred_idx    = np.argmax(predictions)

            if confidence > THRESHOLD:
                prediction = SIGNS[pred_idx]
                pred_color = (0, 255, 0)
            else:
                prediction = '...'
                pred_color = (0, 165, 255)

        h, w, _ = frame.shape

        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, f'Sign: {prediction.upper()}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 3)
        cv2.putText(frame, f'{confidence*100:.1f}%', (w - 120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        bar_len = int((len(sequence) / FRAMES) * 200)
        cv2.rectangle(frame, (20, h - 30), (220, h - 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - 30), (20 + bar_len, h - 10), (0, 255, 0), -1)
        cv2.putText(frame, 'Buffer', (225, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('ISL Real Time Detection', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()