# ovarian_classifier.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# --- Load expression matrix (samples x genes) ---
# For now, placeholder CSV
expr = pd.read_csv("data/expression_matrix.csv", index_col=0)  # samples x genes
labels = pd.read_csv("data/labels.csv", index_col=0)           # cancer vs control

X = expr.values
y = labels["class"].values  # 0 = control, 1 = ovarian cancer

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- Normalize ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Train classifier ---
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
