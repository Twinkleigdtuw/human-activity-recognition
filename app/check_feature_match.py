import numpy as np
import pandas as pd
from pathlib import Path

from predictor import extract_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load raw test windows
test_data = np.load(
    PROJECT_ROOT / "data" / "processed" / "test.npz",
    allow_pickle=True
)

X_test = test_data["X"]

# Load already-generated features
feature_df = pd.read_csv(
    PROJECT_ROOT / "data" / "processed" / "test_features.csv"
)

feature_columns = feature_df.columns[:-1]

# Extract features ourselves
calculated = extract_features(X_test)

# Compare first few rows
saved = feature_df[feature_columns].values

print("=" * 60)
print("FEATURE PIPELINE CHECK")
print("=" * 60)

print("Calculated shape:", calculated.shape)
print("Saved shape:     ", saved.shape)

for i in range(5):
    difference = np.max(
        np.abs(calculated[i] - saved[i])
    )

    print()
    print(f"Window {i + 1}")
    print("Maximum feature difference:", difference)

print()
print("=" * 60)

if np.allclose(calculated, saved):
    print("RESULT: FEATURES MATCH EXACTLY")
else:
    print("RESULT: FEATURES DO NOT MATCH")

print("=" * 60)