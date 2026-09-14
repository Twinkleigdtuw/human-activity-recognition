import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"

train = np.load(DATA_PATH / "train.npz", allow_pickle=True)
test = np.load(DATA_PATH / "test.npz", allow_pickle=True)

X_train = train["X"]
y_train = train["y"]

X_test = test["X"]
y_test = test["y"]

sensor_names = ["Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]

def extract_features(X):
    features = []

    for window in X:
        row = []

        for i in range(6):
            signal = window[:, i]

            mean = np.mean(signal)
            std = np.std(signal)
            minimum = np.min(signal)
            maximum = np.max(signal)
            range_value = maximum - minimum
            median = np.median(signal)
            rms = np.sqrt(np.mean(signal ** 2))

            row.extend([
                mean, std, minimum, maximum,
                range_value, median, rms
            ])

        features.append(row)

    return np.array(features)

X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

feature_names = []

for sensor in sensor_names:
    feature_names.extend([
        sensor + "_mean",
        sensor + "_std",
        sensor + "_min",
        sensor + "_max",
        sensor + "_range",
        sensor + "_median",
        sensor + "_rms"
    ])

train_df = pd.DataFrame(X_train_features, columns=feature_names)
test_df = pd.DataFrame(X_test_features, columns=feature_names)

train_df["activity"] = y_train
test_df["activity"] = y_test

train_df.to_csv(DATA_PATH / "train_features.csv", index=False)
test_df.to_csv(DATA_PATH / "test_features.csv", index=False)

print("Feature engineering complete!")
print("Training features:", X_train_features.shape)
print("Testing features:", X_test_features.shape)
print("Number of features:", len(feature_names))
print("Files saved successfully.")