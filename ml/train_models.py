import pandas as pd

# load processed data
train = pd.read_csv("data/processed/train_features.csv")
test = pd.read_csv("data/processed/test_features.csv")

# separate features and target
X_train = train.drop("activity", axis=1)
y_train = train["activity"]

X_test = test.drop("activity", axis=1)
y_test = test["activity"]

# check dataset
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nActivities:")
print(y_train.value_counts())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# train the model
rf_model.fit(X_train, y_train)

# make predictions on unseen test data
rf_predictions = rf_model.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Results")
print("---------------------")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# train the model
rf_model.fit(X_train, y_train)

# make predictions on test data
rf_predictions = rf_model.predict(X_test)

# evaluate the model
accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Results")
print("---------------------")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# confusion matrix
cm = confusion_matrix(y_test, rf_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=rf_model.classes_
)

disp.plot(xticks_rotation=90)
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig("results/random_forest_confusion_matrix.png")
plt.show()


# feature importance
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# plot top 10 features
feature_importance.head(10).sort_values().plot(kind="barh")
plt.title("Random Forest - Top 10 Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("results/random_forest_feature_importance.png")
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# create SVM model
svm_model = SVC(
    kernel="rbf",
    random_state=42
)

# train SVM
svm_model.fit(X_train_scaled, y_train)

# make predictions
svm_predictions = svm_model.predict(X_test_scaled)

# evaluate
svm_accuracy = accuracy_score(y_test, svm_predictions)

print("\nSVM Results")
print("-----------")
print("Accuracy:", svm_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, svm_predictions))

from sklearn.linear_model import LogisticRegression

# create Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

# train Logistic Regression
logistic_model.fit(X_train_scaled, y_train)

# make predictions
logistic_predictions = logistic_model.predict(X_test_scaled)

# evaluate
logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print("\nLogistic Regression Results")
print("---------------------------")
print("Accuracy:", logistic_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, logistic_predictions))
import joblib

# save the trained Random Forest model
joblib.dump(rf_model, "models/random_forest_model.joblib")

print("\nRandom Forest model saved successfully.")