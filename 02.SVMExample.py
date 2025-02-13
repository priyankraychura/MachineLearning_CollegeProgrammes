import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load the dataset
data = pd.read_csv("loan_approval.csv")  # Replace with your dataset

# Define features (X) and target (y)
X = data.drop(columns=['LoanApproved'])  # Features
y = data['LoanApproved']  # Target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a linear SVM classifier
linear_svm = SVC(kernel='linear', C=1.0, probability=True)
linear_svm.fit(X_train, y_train)

# Make predictions
y_pred = linear_svm.predict(X_test)

# Evaluate the Linear SVM model
print("Accuracy (Linear SVM):", accuracy_score(y_test, y_pred))
print("\nClassification Report (Linear SVM):")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix for Linear SVM
cm_linear = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(cm.shape[1]), labels=["Class 0", "Class 1"])
    plt.yticks(range(cm.shape[0]), labels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.show()

plot_confusion_matrix(cm_linear, "Confusion Matrix - Linear SVM")

# Train a non-linear SVM classifier with RBF kernel
non_linear_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
non_linear_svm.fit(X_train, y_train)

# Make predictions
y_pred_non_linear = non_linear_svm.predict(X_test)

# Evaluate the Non-Linear SVM model
print("Accuracy (Non-Linear SVM):", accuracy_score(y_test, y_pred_non_linear))
print("\nClassification Report (Non-Linear SVM):")
print(classification_report(y_test, y_pred_non_linear, zero_division=0))

# Confusion Matrix for Non-Linear SVM
cm_non_linear = confusion_matrix(y_test, y_pred_non_linear)
plot_confusion_matrix(cm_non_linear, "Confusion Matrix - Non-Linear SVM")

# Model Evaluation (ROC Curve and AUC)
# For Linear SVM
y_probs_linear = linear_svm.predict_proba(X_test)[:, 1]
fpr_linear, tpr_linear, _ = roc_curve(y_test, y_probs_linear)
roc_auc_linear = auc(fpr_linear, tpr_linear)

# For Non-Linear SVM
y_probs_non_linear = non_linear_svm.predict_proba(X_test)[:, 1]
fpr_non_linear, tpr_non_linear, _ = roc_curve(y_test, y_probs_non_linear)
roc_auc_non_linear = auc(fpr_non_linear, tpr_non_linear)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_linear, tpr_linear, color='blue', label=f'Linear SVM (AUC = {roc_auc_linear:.2f})')
plt.plot(fpr_non_linear, tpr_non_linear, color='green', label=f'Non-Linear SVM (AUC = {roc_auc_non_linear:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Visualizing Decision Boundaries (for 2D data only)
# Note: Decision boundaries can only be plotted for 2D feature spaces.

# Create a synthetic dataset with 2 features for visualization
from sklearn.datasets import make_classification
X_vis, y_vis = make_classification(n_samples=500, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)
X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(X_vis, y_vis, test_size=0.3, random_state=42)

X_vis_train = scaler.fit_transform(X_vis_train)
X_vis_test = scaler.transform(X_vis_test)

# Train a non-linear SVM for visualization
svm_vis = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_vis.fit(X_vis_train, y_vis_train)

# Plot decision boundaries
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(X_vis_train, y_vis_train, svm_vis, "Decision Boundary - Non-Linear SVM")
