"""
Check how well the posture model works on people it has never seen.

train.py uses a random 80/20 split. Each recording is a run of nearly
identical frames, so a random split puts frames from the same recording
in both training and testing, and the score comes out too high.

This script keeps the features and Random Forest settings from train.py
and changes only the split:

  1. Random split (same as train.py), as the baseline.
  2. Leave one person out: train on everyone else, test on that person.
  3. Group by recording: no recording appears in both training and testing.

It does not save a model. train.py still produces posture_model.pkl.

Run from the repo root or from train/:
    python train/evaluate_by_person.py
"""

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")

FEATURES = ["neck_angle", "spine_angle", "forward_head_offset", "shoulder_tilt"]

# Who is in each recording. The CSVs do not store this, so it comes from the
# file timestamps: the recordings fall into three sessions with long gaps
# between them. EDIT THIS if it does not match who actually sat for each one.
# Each entry is (first timestamp in the session, person name). A recording
# belongs to the most recent session that started at or before it.
PERSON_SESSIONS = [
    ("20260616_194900", "person_1"),  # 19:49 - 19:54, plus one bad at 20:02
    ("20260616_202000", "person_2"),  # 20:20 - 20:28
    ("20260616_203000", "person_3"),  # 20:31 - 20:35
]


def person_for(timestamp):
    """Return the person for a recording timestamp like 20260616_194918."""
    person = None
    for start, name in PERSON_SESSIONS:
        if timestamp >= start:
            person = name
    if person is None:
        raise ValueError(f"No person session covers recording {timestamp}")
    return person


def load_dataset():
    dfs = []
    for folder, posture in (("good_posture", 0), ("bad_posture", 1)):
        folder_path = os.path.join(RUNS_DIR, folder)
        for file in sorted(os.listdir(folder_path)):
            if not file.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(folder_path, file))
            # "bad_20260616_194926.csv" -> "20260616_194926"
            timestamp = file[:-4].split("_", 1)[1]
            df["posture"] = posture
            df["recording"] = file
            # Use the person column if collect.py wrote one, else infer it.
            if "person" not in df.columns or df["person"].isna().all():
                df["person"] = person_for(timestamp)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def new_model():
    # Same settings as train.py
    return RandomForestClassifier(n_estimators=100, random_state=42)


def f1_bad(y_true, y_pred):
    # posture 1 = bad, so this is the F1 score for catching bad posture
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)


def main():
    dataset = load_dataset()
    X = dataset[FEATURES]
    y = dataset["posture"]

    print("Frames per person (0 = good, 1 = bad):")
    print(pd.crosstab(dataset["person"], dataset["posture"]))
    print()

    # 1. Baseline: random split, exactly as in train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = new_model().fit(X_train, y_train)
    random_f1 = f1_bad(y_test, model.predict(X_test))
    print(f"1. Random split (train.py)       F1 = {random_f1:.3f}")
    print()

    # 2. Leave one person out
    print("2. Leave one person out (train on the others, test on this person)")
    groups = dataset["person"]
    scores = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        model = new_model().fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        y_true = y.iloc[test_idx]
        s = f1_bad(y_true, pred)
        acc = (pred == y_true.values).mean()
        scores.append(s)
        print(
            f"   {groups.iloc[test_idx[0]]:<10} F1 = {s:.3f}   "
            f"accuracy = {acc:.3f}   ({len(test_idx)} frames)"
        )
    person_f1 = sum(scores) / len(scores)
    print(f"   Average F1 on unseen people   F1 = {person_f1:.3f}")
    print()

    # 3. Group by recording: no recording is split across train and test
    print("3. Grouped by recording (5 folds, no recording in both sets)")
    rec_groups = dataset["recording"]
    rec_scores = []
    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=5).split(X, y, rec_groups), start=1
    ):
        model = new_model().fit(X.iloc[train_idx], y.iloc[train_idx])
        s = f1_bad(y.iloc[test_idx], model.predict(X.iloc[test_idx]))
        rec_scores.append(s)
        print(f"   fold {fold}  F1 = {s:.3f}")
    rec_f1 = sum(rec_scores) / len(rec_scores)
    print(f"   Average F1 on unseen recordings F1 = {rec_f1:.3f}")
    print()

    print("Summary")
    print(f"   Random split:      {random_f1:.3f}")
    print(f"   Unseen recordings: {rec_f1:.3f}")
    print(f"   Unseen people:     {person_f1:.3f}")


if __name__ == "__main__":
    main()
