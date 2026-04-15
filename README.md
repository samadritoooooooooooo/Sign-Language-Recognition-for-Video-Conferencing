# SL Meet — Sign Language Recognition for Video Conferencing

A real-time Sign Language (SL) recognition system built with MediaPipe, OpenCV, and LSTM neural networks. Designed to help deaf and hard-of-hearing individuals communicate effectively in video conferencing environments like Google Meet.

---

## Overview

This system recognizes 6 SL gestures commonly used in meeting contexts and displays them as on-screen text in real time. A two-layer LSTM model trained on hand landmark sequences achieves **94.44% test accuracy**.

| Gesture     | Meaning              |
| ----------- | -------------------- |
| ✋ Yes      | Agreement            |
| 🤚 No       | Disagreement         |
| ☝️ Question | I have a question    |
| 🔄 Repeat   | Can you repeat that? |
| 👍 OK       | Understood           |
| 🖐️ Wait     | Please wait          |

---

## Pipeline

```
Webcam → OpenCV → MediaPipe (21 landmarks × xyz = 63 values)
       → 30-frame sequences → LSTM → Predicted sign + confidence %
```

---

## Tech Stack

- **MediaPipe** — hand landmark detection
- **OpenCV** — webcam feed and UI
- **TensorFlow / Keras** — LSTM model training and inference
- **NumPy** — landmark data storage (`.npy` format)
- **Python 3.12**

---

## Project Structure

```
SL-Meet/
├── collect.py          # Data collection script (MediaPipe + webcam)
├── train.py            # LSTM model training
├── test.py             # Real-time inference
├── sl_model.keras     # Trained model
├── SL_Data/           # Dataset (120 sequences × 30 frames per sign)
│   ├── yes/
│   ├── no/
│   ├── question/
│   ├── repeat/
│   ├── ok/
│   └── wait/
└── hand_landmarker.task  # MediaPipe model file
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/SL-Meet.git
cd SL-Meet

# Create a virtual environment (Python 3.12 recommended)
python3.12 -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
pip install mediapipe tensorflow opencv-python numpy scikit-learn matplotlib
```

Download the MediaPipe hand landmarker model and place it in the project root:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## Usage

**1. Collect data**

```bash
python collect.py
# Set CURRENT_SIGN = 'yes' (or any sign) in the script
# Press SPACE to start, pauses every 40 clips for next person
```

**2. Train the model**

```bash
python train.py
# Trains LSTM on collected data, saves sl_model.keras
```

**3. Run real-time detection**

```bash
python test.py
# Shows live webcam feed with predicted sign and confidence score
# Press Q to quit
```

---

## Model

| Parameter        | Value                     |
| ---------------- | ------------------------- |
| Architecture     | 2-layer LSTM              |
| Input shape      | (30 frames, 63 landmarks) |
| Training samples | 576                       |
| Test accuracy    | **94.44%**                |
| Optimizer        | Adam                      |
| Regularization   | Dropout (0.2)             |

---

## Dataset

Collected via webcam using MediaPipe hand landmark extraction. Each sign has 120 sequences of 30 frames recorded by 3 people to ensure variety.

- Total clips: 720 (120 × 6 signs)
- Features per frame: 63 (21 landmarks × x, y, z)
- Input shape to LSTM: (30, 63)

## References

Natarajan, B., Rajalakshmi, E., Elakkiya, R., Kotecha, K., Abraham, A., Gabralla, L. A., & Subramaniyaswamy, V. (2022). Development of an end-to-end deep learning framework for sign language recognition, translation, and video generation. IEEE Access, 10, 104358–104370. https://doi.org/10.1109/ACCESS.2022.3210543

---

## Team

1. Mohammed Afzal Asim (2023UCS1613) (Netaji Subhas University of Technology).

2. Samadrito Chakraborty(2023UCS1621) (Netaji Subhas University of Technology).
