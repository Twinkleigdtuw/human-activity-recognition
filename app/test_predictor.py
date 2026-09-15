import numpy as np
from pathlib import Path

from predictor import predict_batch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# Load REAL test data
# --------------------------------------------------

test_data = np.load(
    PROJECT_ROOT / "data" / "processed" / "test.npz",
    allow_pickle=True
)

X_test = test_data["X"]
y_test = test_data["y"]


# --------------------------------------------------
# Batch prediction
# --------------------------------------------------

print("=" * 60)
print("FULL PREDICTOR TEST")
print("=" * 60)

predictions, confidences, probabilities = predict_batch(X_test)


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

correct = np.sum(predictions == y_test)

accuracy = correct / len(y_test)


print(f"Total samples : {len(y_test)}")
print(f"Correct       : {correct}")
print(f"Incorrect     : {len(y_test) - correct}")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Accuracy (%)  : {accuracy:.2%}")


# --------------------------------------------------
# First 10 samples
# --------------------------------------------------

print()
print("First 10 predictions:")
print("-" * 60)

for i in range(10):

    print(
        f"{i + 1:2d}. "
        f"Actual: {y_test[i]:20s} | "
        f"Predicted: {predictions[i]:20s} | "
        f"Confidence: {confidences[i]:.2%}"
    )


print("=" * 60)