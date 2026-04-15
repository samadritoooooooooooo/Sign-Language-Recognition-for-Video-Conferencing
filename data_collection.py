import cv2
import mediapipe as mp
import numpy as np
import os


SIGNS = ['yes', 'no', 'question', 'repeat', 'ok', 'wait']
SEQUENCES = 120
FRAMES    = 30
PAUSE_EVERY = 40
DATA_DIR  = 'ISL_Data'
MODEL_PATH = 'C:\\Users\\samth\\Downloads\\Indian-sign-language-using-OpenCV-main\\Indian-sign-language-using-OpenCV-main\\hand_landmarker.task'


for sign in SIGNS:
    for seq in range(SEQUENCES):
        os.makedirs(os.path.join(DATA_DIR, sign, str(seq)), exist_ok=True)


CURRENT_SIGN = 'ok'



def get_start_sequence(sign):

    sign_path = os.path.join(DATA_DIR, sign)
    for seq in range(SEQUENCES):
        seq_path = os.path.join(sign_path, str(seq))
        if not os.path.exists(os.path.join(seq_path, '29.npy')):
            return seq
    return SEQUENCES

start_seq = get_start_sequence(CURRENT_SIGN)

if start_seq >= SEQUENCES:
    print(f" {CURRENT_SIGN.upper()} already has all {SEQUENCES} clips recorded!")
else:
    print(f" {CURRENT_SIGN.upper()} — resuming from clip {start_seq + 1}/{SEQUENCES}")


BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
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


cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:


    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'SIGN: {CURRENT_SIGN.upper()}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        cv2.putText(frame, f'Starting from clip {start_seq + 1}/{SEQUENCES}', (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, 'Press SPACE when ready', (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('ISL Data Collection', frame)
        if cv2.waitKey(10) & 0xFF == ord(' '):
            break

    for seq in range(start_seq, SEQUENCES):


        if seq != start_seq and (seq - start_seq) % PAUSE_EVERY == 0:
            while True:
                ret, frame = cap.read()
                cv2.putText(frame, 'NEXT PERSON UP!', (80, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                cv2.putText(frame, f'Clips done: {seq}/{SEQUENCES}', (80, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, 'Press SPACE when ready', (80, 320),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('ISL Data Collection', frame)
                if cv2.waitKey(10) & 0xFF == ord(' '):
                    break


        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            cv2.putText(frame, f'{CURRENT_SIGN.upper()}  |  Clip {seq+1}/{SEQUENCES}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, str(countdown), (300, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 8)
            cv2.imshow('ISL Data Collection', frame)
            cv2.waitKey(500)


        for frame_num in range(FRAMES):
            ret, frame = cap.read()

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_image)

            draw_hands(frame, result)

            cv2.putText(frame, f'{CURRENT_SIGN.upper()}  |  Clip {seq+1}/{SEQUENCES}  |  Frame {frame_num+1}/{FRAMES}',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('ISL Data Collection', frame)
            cv2.waitKey(33)

            landmarks = extract_landmarks(result)
            save_path = os.path.join(DATA_DIR, CURRENT_SIGN, str(seq), f'{frame_num}.npy')
            np.save(save_path, landmarks)

        print(f' {CURRENT_SIGN} — clip {seq+1}/{SEQUENCES} saved')

    print(f'\n {CURRENT_SIGN.upper()} DONE!\n')

cap.release()
cv2.destroyAllWindows()
print("Session complete!")