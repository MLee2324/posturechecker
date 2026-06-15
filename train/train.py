import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Go up one folder
good_dir = "../runs/good_posture"
bad_dir = "../runs/bad_posture"

dfs = []

# Load good posture files
for file in os.listdir(good_dir):
    if file.endswith(".csv"):
        path = os.path.join(good_dir, file)

        df = pd.read_csv(path)
        df["posture"] = 0

        dfs.append(df)

# Load bad posture files
for file in os.listdir(bad_dir):
    if file.endswith(".csv"):
        path = os.path.join(bad_dir, file)

        df = pd.read_csv(path)
        df["posture"] = 1

        dfs.append(df)

# Combine into one dataframe
dataset = pd.concat(dfs, ignore_index=True)

print(dataset["posture"].value_counts())
print(dataset.head())
print(dataset.tail())

# train the random forest classifier model
X = dataset[
    [
        "neck_angle",
        "spine_angle",
        "forward_head_offset",
        "shoulder_tilt"
    ]
]

y = dataset["posture"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier(n_estimators=100, random_state = 42)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

for feature, importance in zip(X.columns, model.feature_importances_):
    print(feature, importance)

# load to dump to use in main.py
joblib.dump(model, "posture_model.pkl")