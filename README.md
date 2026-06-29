# Posture Checker

A real-time posture monitoring system that uses MediaPipe Pose and a Random Forest classifier to detect good and bad sitting posture from a webcam feed.

---

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Collect Training Data

Run:

```bash
python collect.py
```

### Controls

```text
G = Record Good Posture
B = Record Bad Posture
S = Stop and Save
Q = Quit
```

The recorded CSV files will be saved to:

```text
runs/good_posture/
runs/bad_posture/
```

---

## Train the Model

Run:

```bash
python train/train.py
```

This script will:

- Load all collected CSV files
- Combine the dataset
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
- Detect body landmarks using MediaPipe Pose
- Extract posture features
- Load the trained Random Forest model
- Predict posture in real time
- Display posture status on the screen
- Play an audio alert while poor posture is detected

### Controls

```text
Q = Quit
```

---

## Feature Definitions

### Neck Angle

Angle formed by:

```text
Ear → Neck → Shoulder Midpoint
```

Measures the alignment of the head relative to the shoulders.

### Forward Head Offset

Horizontal distance between:

```text
Ear and Neck
```

Larger values indicate that the head is positioned farther forward.

### Shoulder Tilt

Vertical difference between:

```text
Left Shoulder and Right Shoulder
```

Measures shoulder imbalance or leaning.

---

## Troubleshooting

### Webcam does not open

- Make sure no other application is using the webcam.
- Verify that your operating system has granted camera permissions to Python or your IDE.

### MediaPipe installation issues

Try reinstalling the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you are using macOS, Python 3.9 or 3.10 is recommended for the best MediaPipe compatibility.

### Model file not found

If you receive an error such as:

```text
FileNotFoundError: posture_model.pkl
```

Train the model first:

```bash
python train/train.py
```

This will generate:

```text
train/posture_model.pkl
```

### Poor prediction accuracy

- Collect additional training samples.
- Record examples from different sitting positions and lighting conditions.
- Keep the camera positioned consistently during both training and testing.

### No posture detection

- Ensure your upper body is visible in the camera frame.
- Avoid excessive occlusion of the face or shoulders.
- Make sure the environment has sufficient lighting.

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

---

## Author

Matthew Lee