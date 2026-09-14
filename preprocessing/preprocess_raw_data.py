import numpy as np
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_HAR" / "RawData"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

print("Project root:", PROJECT_ROOT)
print("Raw data path:", RAW_DATA_PATH)
print("Processed data path:", PROCESSED_DATA_PATH)

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
print("Processed data folder is ready.")

# Check raw data files
print("\nRaw data files:")

for file in sorted(RAW_DATA_PATH.iterdir()):
    print(file.name)

# Check accelerometer and gyroscope files
acc_files = sorted(RAW_DATA_PATH.glob("acc_*.txt"))
gyro_files = sorted(RAW_DATA_PATH.glob("gyro_*.txt"))

print("\nNumber of accelerometer files:", len(acc_files))
print("Number of gyroscope files:", len(gyro_files))

print("\nFirst 5 accelerometer files:")
for file in acc_files[:5]:
    print(file.name)

print("\nFirst 5 gyroscope files:")
for file in gyro_files[:5]:
    print(file.name)

# Load one accelerometer and gyroscope file
acc_file = acc_files[0]
gyro_file = gyro_files[0]

print("\nLoading:")
print("Accelerometer:", acc_file.name)
print("Gyroscope:", gyro_file.name)

acc_data = np.loadtxt(acc_file)
gyro_data = np.loadtxt(gyro_file)

print("\nAccelerometer shape:", acc_data.shape)
print("Gyroscope shape:", gyro_data.shape)

print("\nFirst 5 accelerometer samples:")
print(acc_data[:5])

print("\nFirst 5 gyroscope samples:")
print(gyro_data[:5])

# Load labels
LABELS_PATH = RAW_DATA_PATH / "labels.txt"
labels = np.loadtxt(LABELS_PATH)

print("\nLabels shape:", labels.shape)
print("\nFirst 10 labels:")
print(labels[:10])

# Activity mapping
ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
    7: "STAND_TO_SIT",
    8: "SIT_TO_STAND",
    9: "SIT_TO_LIE",
    10: "LIE_TO_SIT",
    11: "STAND_TO_LIE",
    12: "LIE_TO_STAND"
}

print("\nActivity mapping:")

for activity_id, activity_name in ACTIVITY_NAMES.items():
    print(activity_id, "->", activity_name)

# Process all experiments and users
X_all = []
y_all = []

print("\nProcessing all sensor recordings...")

for acc_file in acc_files:

    parts = acc_file.stem.split("_")

    experiment = int(parts[1].replace("exp", ""))
    user = int(parts[2].replace("user", ""))

    gyro_filename = f"gyro_exp{experiment:02d}_user{user:02d}.txt"
    gyro_file = RAW_DATA_PATH / gyro_filename

    if not gyro_file.exists():
        print("Gyroscope file missing:", gyro_filename)
        continue

    acc_data = np.loadtxt(acc_file)
    gyro_data = np.loadtxt(gyro_file)

    sensor_data = np.hstack((acc_data, gyro_data))

    experiment_labels = labels[
        (labels[:, 0] == experiment) &
        (labels[:, 1] == user)
    ]

    print(f"\nExperiment {experiment:02d} | User {user:02d}")
    print("Sensor shape:", sensor_data.shape)
    print("Labeled segments:", len(experiment_labels))

    # Process each labeled segment
    for row in experiment_labels:

        activity_id = int(row[2])
        start_sample = int(row[3])
        end_sample = int(row[4])

        activity_name = ACTIVITY_NAMES[activity_id]

        start_index = start_sample - 1
        end_index = end_sample

        segment = sensor_data[start_index:end_index]

        X_all.append(segment)
        y_all.extend([activity_name] * len(segment))

# Combine all recordings
X_all = np.vstack(X_all)
y_all = np.array(y_all)

print("\nFinal dataset:")
print("X shape:", X_all.shape)
print("y shape:", y_all.shape)

# Save processed dataset
output_file = PROCESSED_DATA_PATH / "har_sensor_data.npz"

np.savez_compressed(
    output_file,
    X=X_all,
    y=y_all
)

print("\nProcessed dataset saved to:")
print(output_file)

# Show activity distribution
print("\nActivity distribution:")

activity_counts = pd.Series(y_all).value_counts()
print(activity_counts)