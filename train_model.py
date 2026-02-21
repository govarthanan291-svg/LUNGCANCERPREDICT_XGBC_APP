import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve

df = pd.read_csv("survey_lung_cancer.csv")
df.columns = df.columns.str.strip()

le_gender = LabelEncoder()
df['GENDER'] = le_gender.fit_transform(df['GENDER'])
le_target = LabelEncoder()
df['LUNG_CANCER'] = le_target.fit_transform(df['LUNG_CANCER'])

binary_cols = [c for c in df.columns if c not in ['GENDER','AGE','LUNG_CANCER']]
for c in binary_cols:
    df[c] = df[c] - 1

X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER']
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=4,
    subsample=0.8, min_samples_split=10, random_state=42
)
model.fit(X_train, y_train)
y_pred   = model.predict(X_test)
y_proba  = model.predict_proba(X_test)[:,1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f}")
print(classification_report(y_test, y_pred))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
fpr, tpr, _ = roc_curve(y_test, y_proba)
fi_sorted = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)

artifacts = {
    "feature_names": feature_names,
    "binary_cols": binary_cols,
    "model_comparison": {"XGBoost (GBC)": {"acc": round(acc*100,2), "auc": round(auc,4), "cv": round(cv_scores.mean()*100,2)}},
    "cv_scores": cv_scores.tolist(), "cv_mean": float(cv_scores.mean()), "cv_std": float(cv_scores.std()),
    "accuracy": float(acc), "auc": float(auc),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
    "feature_importances": dict(zip(feature_names, model.feature_importances_.tolist())),
    "fi_sorted": fi_sorted,
    "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist(),
}

with open("lung_xgb_model.pkl","wb") as f: pickle.dump(model, f)
with open("lung_le_gender.pkl","wb") as f: pickle.dump(le_gender, f)
with open("lung_le_target.pkl","wb") as f: pickle.dump(le_target, f)
with open("lung_artifacts.pkl","wb") as f: pickle.dump(artifacts, f)
with open("lung_feature_names.pkl","wb") as f: pickle.dump(feature_names, f)
print("✅ All artifacts saved!")
