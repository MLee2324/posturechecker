# Posture Checker

A real-time posture monitoring system that uses MediaPipe Pose and a Random Forest classifier to detect good and bad sitting posture from a webcam feed.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Collect Training Data

Run:

```bash
python collect.py
```

Controls:

```text
G = Record Good Posture
B = Record Bad Posture
S = Stop and Save
Q = Quit
```

CSV files will be saved to:

```text
runs/good_posture/
runs/bad_posture/
```

---

## Train Model

Run:

```bash
python train/train.py
```

This will:

- Load all CSV files
- Train a Random Forest classifier
- Evaluate model performance
- Save the trained model as:

```text
train/posture_model.pkl
```

---

## Run Live Posture Detection

Run:

```bash
python main.py
```

The application will:

- Open the webcam
- Extract posture features
- Load the trained model
- Predict Good or Bad posture in real time
- Display posture status on screen
- Play an audio alert when poor posture is detected

Press:

```text
Q = Quit
```

---

## Feature Definitions

### Neck Angle

Angle formed by:

```text
Ear → Neck → Shoulder
```

Used to measure head alignment.

### Spine Angle

Angle formed by:

```text
Ear → Shoulder → Hip
```

Used to estimate upper-body curvature.

### Forward Head Offset

Horizontal distance between:

```text
Ear and Neck
```

Larger values indicate the head is positioned farther forward.

### Shoulder Tilt

Vertical difference between:

```text
Left Shoulder and Right Shoulder
```

Used to measure shoulder asymmetry.

---

## Project Structure

```text
posturechecker/
│
├── main.py
├── collect.py
├── train/
│   ├── train.py
│   └── posture_model.pkl
│
├── drawers/
│   └── visuals.py
│
├── runs/
│   ├── good_posture/
│   └── bad_posture/
│
└── requirements.txt
```