import numpy as np
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "UCI_HAR" / "RawData"
)

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DATA_PATH / "har_windows.npz"

# Window settings
SAMPLING_RATE = 50
WINDOW_SIZE = 128
STEP_SIZE = 64

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

print("Creating windows...")
print("Sampling rate:", SAMPLING_RATE, "Hz")
print("Window size:", WINDOW_SIZE, "samples")
print("Window duration:", WINDOW_SIZE / SAMPLING_RATE, "seconds")
print("Step size:", STEP_SIZE, "samples")
print()

# Load labels
LABELS_PATH = RAW_DATA_PATH / "labels.txt"
labels = np.loadtxt(LABELS_PATH)

print("Labels loaded:", labels.shape)
print()

# Find sensor files
acc_files = sorted(RAW_DATA_PATH.glob("acc_*.txt"))
gyro_files = sorted(RAW_DATA_PATH.glob("gyro_*.txt"))

print("Accelerometer files:", len(acc_files))
print("Gyroscope files:", len(gyro_files))
print()

# Storage for windows
X_windows = []
y_windows = []
user_ids = []
experiment_ids = []

# Process each recording
for acc_file, gyro_file in zip(acc_files, gyro_files):

    # Extract experiment and user
    parts = acc_file.stem.split("_")

    experiment = int(parts[1].replace("exp", ""))
    user = int(parts[2].replace("user", ""))

    # Load and combine sensor data
    acc_data = np.loadtxt(acc_file)
    gyro_data = np.loadtxt(gyro_file)

    sensor_data = np.hstack((acc_data, gyro_data))

    # Find labels for this recording
    recording_labels = labels[
        (labels[:, 0] == experiment) &
        (labels[:, 1] == user)
    ]

    print(
        f"Experiment {experiment:02d} | "
        f"User {user:02d} | "
        f"Samples: {len(sensor_data)} | "
        f"Segments: {len(recording_labels)}"
    )

    # Process each labeled segment
    for row in recording_labels:

        activity_id = int(row[2])
        start_sample = int(row[3])
        end_sample = int(row[4])

        activity_name = ACTIVITY_NAMES[activity_id]

        # Convert dataset indexing to Python indexing
        start_index = start_sample - 1
        end_index = end_sample

        segment = sensor_data[start_index:end_index]

        # Create windows inside the segment
        start = 0

        while start + WINDOW_SIZE <= len(segment):

            end = start + WINDOW_SIZE
            window = segment[start:end]

            # Store window and metadata
            X_windows.append(window)
            y_windows.append(activity_name)
            user_ids.append(user)
            experiment_ids.append(experiment)

            start += STEP_SIZE

# Convert lists to NumPy arrays
X_windows = np.array(X_windows)
y_windows = np.array(y_windows)
user_ids = np.array(user_ids)
experiment_ids = np.array(experiment_ids)

# Display results
print()
print("Windowing complete!")

print("X shape:", X_windows.shape)
print("y shape:", y_windows.shape)
print("User IDs shape:", user_ids.shape)
print("Experiment IDs shape:", experiment_ids.shape)

print()
print("One window shape:", X_windows[0].shape)

print()
print("First 10 window labels:")
print(y_windows[:10])

print()
print("First 10 user IDs:")
print(user_ids[:10])

print()
print("First 10 experiment IDs:")
print(experiment_ids[:10])

# Activity distribution
print()
print("Activity distribution:")

unique_labels, counts = np.unique(
    y_windows,
    return_counts=True
)

for label, count in zip(unique_labels, counts):
    print(f"{label}: {count}")

# User distribution
print()
print("User distribution:")

unique_users, user_counts = np.unique(
    user_ids,
    return_counts=True
)

for user, count in zip(unique_users, user_counts):
    print(f"User {user:02d}: {count} windows")

# Save dataset
np.savez_compressed(
    OUTPUT_FILE,
    X=X_windows,
    y=y_windows,
    user_ids=user_ids,
    experiment_ids=experiment_ids
)

print()
print("Dataset saved to:")
print(OUTPUT_FILE)