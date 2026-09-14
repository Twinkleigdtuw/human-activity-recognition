import numpy as np
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DATA_PATH / "har_windows.npz"
TRAIN_FILE = PROCESSED_DATA_PATH / "train.npz"
TEST_FILE = PROCESSED_DATA_PATH / "test.npz"

# Load windowed data
print("Loading windowed data...")

data = np.load(INPUT_FILE)

X = data["X"]
y = data["y"]
user_ids = data["user_ids"]
experiment_ids = data["experiment_ids"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("User IDs shape:", user_ids.shape)
print("Experiment IDs shape:", experiment_ids.shape)
print()

# Check available users
unique_users = np.unique(user_ids)

print("Users available:")
print(unique_users)
print()
print("Total users:", len(unique_users))
print()

# Define user-wise train/test split
TRAIN_USERS = np.arange(1, 25)
TEST_USERS = np.arange(25, 31)

print("Training users:")
print(TRAIN_USERS)
print()

print("Testing users:")
print(TEST_USERS)
print()

# Create train/test masks
train_mask = np.isin(user_ids, TRAIN_USERS)
test_mask = np.isin(user_ids, TEST_USERS)

# Create training data
X_train = X[train_mask]
y_train = y[train_mask]
train_user_ids = user_ids[train_mask]
train_experiment_ids = experiment_ids[train_mask]

# Create testing data
X_test = X[test_mask]
y_test = y[test_mask]
test_user_ids = user_ids[test_mask]
test_experiment_ids = experiment_ids[test_mask]

# Display split results
print("Train / Test Split")
print()
print("Training data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print()

print("Testing data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print()

# Verify user separation
train_unique_users = np.unique(train_user_ids)
test_unique_users = np.unique(test_user_ids)

print("Training users actually present:")
print(train_unique_users)
print()

print("Testing users actually present:")
print(test_unique_users)
print()

# Check for user overlap
overlap = np.intersect1d(
    train_unique_users,
    test_unique_users
)

print("Users appearing in BOTH sets:")
print(overlap)
print()

if len(overlap) == 0:
    print("✓ No user overlap!")
    print("✓ Train and test users are completely separate.")
else:
    print("WARNING: User overlap detected!")

# Check activity distribution
print()
print("Training activity distribution:")

train_labels, train_counts = np.unique(
    y_train,
    return_counts=True
)

for label, count in zip(train_labels, train_counts):
    print(f"{label}: {count}")

print()
print("Testing activity distribution:")

test_labels, test_counts = np.unique(
    y_test,
    return_counts=True
)

for label, count in zip(test_labels, test_counts):
    print(f"{label}: {count}")

# Save training data
np.savez_compressed(
    TRAIN_FILE,
    X=X_train,
    y=y_train,
    user_ids=train_user_ids,
    experiment_ids=train_experiment_ids
)

# Save testing data
np.savez_compressed(
    TEST_FILE,
    X=X_test,
    y=y_test,
    user_ids=test_user_ids,
    experiment_ids=test_experiment_ids
)

# Final message
print()
print("Split complete!")
print()

print("Training dataset saved to:")
print(TRAIN_FILE)

print()

print("Testing dataset saved to:")
print(TEST_FILE)