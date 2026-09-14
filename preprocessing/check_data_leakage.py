import numpy as np
from pathlib import Path
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATH = PROJECT_ROOT / "data" / "processed"
# Load data
train = np.load(PATH / "train.npz", allow_pickle=True)
test = np.load(PATH / "test.npz", allow_pickle=True)
# Find unique users
train_users = np.unique(train["user_ids"])
test_users = np.unique(test["user_ids"])
# Check user overlap
overlap = np.intersect1d(train_users, test_users)
# Final result
print("Training users:", train_users)
print("Testing users:", test_users)
print("Common users:", overlap)

if len(overlap) == 0:
    print("✓ No user overlap - no subject-level leakage.")
else:
    print("✗ User overlap detected.")