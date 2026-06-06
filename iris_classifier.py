"""
Iris Flower Classification
Compares KNN, SVM, and Decision Tree classifiers with full EDA and visualisation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── Load data ─────────────────────────────────────────────────────────────────
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
target_names = iris.target_names

print("Dataset shape:", X.shape)
print("\nClass distribution:\n", y.value_counts())
print("\nFirst 5 rows:\n", X.head())
print("\nStatistics:\n", X.describe())

# ── EDA: Pairplot ─────────────────────────────────────────────────────────────
df = X.copy()
df["species"] = [target_names[i] for i in y]

sns.pairplot(df, hue="species", palette="Set2", diag_kind="kde")
plt.suptitle("Iris Dataset — Feature Pairplot", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pairplot.png")

# ── EDA: Correlation heatmap ──────────────────────────────────────────────────
plt.figure(figsize=(6, 5))
sns.heatmap(X.corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()
print("Saved correlation_heatmap.png")

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    "K-Nearest Neighbours (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine":     SVC(kernel="rbf", C=1.0, random_state=42),
    "Decision Tree":              DecisionTreeClassifier(max_depth=4, random_state=42),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n" + "="*55)
print("MODEL COMPARISON (5-Fold Cross-Validation)")
print("="*55)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=kf, scoring="accuracy")
    model.fit(X_train_sc, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test_sc))
    results[name] = {"cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(), "test_acc": test_acc}
    print(f"\n{name}")
    print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

# ── Best model detailed report ────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["test_acc"])
best_model = models[best_name]
y_pred = best_model.predict(X_test_sc)

print(f"\n{'='*55}")
print(f"BEST MODEL: {best_name}")
print("="*55)
print(classification_report(y_test, y_pred, target_names=target_names))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved confusion_matrix.png")

print("\nDone. All outputs saved.")
