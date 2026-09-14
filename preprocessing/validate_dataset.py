import numpy as np
from pathlib import Path

# 1. PROJECT PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.npz"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.npz"

# 2. LOAD DATA
train = np.load(TRAIN_PATH, allow_pickle=True)
test = np.load(TEST_PATH, allow_pickle=True)

print("=" * 60)
print("DATASET VALIDATION")
print("=" * 60)

print("\nTrain files:", train.files)
print("Test files:", test.files)

# 3. LOAD X AND y
X_train = train["X"]
y_train = train["y"]

X_test = test["X"]
y_test = test["y"]

print("\nTrain X shape:", X_train.shape)
print("Train y shape:", y_train.shape)

print("Test X shape:", X_test.shape)
print("Test y shape:", y_test.shape)

# 4. CHECK FOR NaN
print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)

print("NaN values in training data:", np.isnan(X_train).sum())
print("NaN values in testing data:", np.isnan(X_test).sum())

# 5. CHECK FOR INFINITY
print("\n" + "=" * 60)
print("INFINITY CHECK")
print("=" * 60)

print("Infinite values in training data:",
      np.isinf(X_train).sum())

print("Infinite values in testing data:",
      np.isinf(X_test).sum())

# 6. CHECK LABELS
print("\n" + "=" * 60)
print("LABEL CHECK")
print("=" * 60)

print("\nTraining classes:")

unique_train, counts_train = np.unique(
    y_train,
    return_counts=True
)

for label, count in zip(unique_train, counts_train):
    print(f"{label}: {count}")


print("\nTesting classes:")

unique_test, counts_test = np.unique(
    y_test,
    return_counts=True
)

for label, count in zip(unique_test, counts_test):
    print(f"{label}: {count}")

# 7. CHECK LABEL CONSISTENCY
print("\n" + "=" * 60)
print("LABEL CONSISTENCY")
print("=" * 60)

train_classes = set(unique_train)
test_classes = set(unique_test)

print("Classes only in training:",
      train_classes - test_classes)

print("Classes only in testing:",
      test_classes - train_classes)

print("Classes in both:",
      train_classes & test_classes)

# 8. CHECK SENSOR VALUE RANGE
print("\n" + "=" * 60)
print("SENSOR VALUE RANGE")
print("=" * 60)

sensor_names = [
    "Acc_X",
    "Acc_Y",
    "Acc_Z",
    "Gyro_X",
    "Gyro_Y",
    "Gyro_Z"
]

for i, sensor in enumerate(sensor_names):

    train_sensor = X_train[:, :, i]
    test_sensor = X_test[:, :, i]

    print(
        f"\n{sensor}:"
        f"\n  Train min = {train_sensor.min():.6f}"
        f"\n  Train max = {train_sensor.max():.6f}"
        f"\n  Test min  = {test_sensor.min():.6f}"
        f"\n  Test max  = {test_sensor.max():.6f}"
    )

# 9. FINAL SUMMARY
print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)

if np.isnan(X_train).sum() == 0 and np.isnan(X_test).sum() == 0:
    print("✓ No NaN values found")
else:
    print("✗ NaN values found")

if np.isinf(X_train).sum() == 0 and np.isinf(X_test).sum() == 0:
    print("✓ No infinite values found")
else:
    print("✗ Infinite values found")

if train_classes == test_classes:
    print("✓ Train/test contain the same activity classes")
else:
    print("⚠ Train/test activity classes differ")