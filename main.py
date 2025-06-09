import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

# === Paths ===
DATA_PATH = r'c:\Users\akhda\Desktop\dry_bean_ml_project\data\Dry_Bean_Dataset.xlsx'
MODELS_PATH = 'models/'
REPORTS_PATH = 'results/classification_reports/'
CM_PATH = 'results/confusion_matrices/'
ACCURACY_CSV_PATH = 'results/accuracy_scores.csv'

# === Ensure directories exist ===
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)
os.makedirs(CM_PATH, exist_ok=True)

# === 1. Load Dataset ===
df = pd.read_excel(DATA_PATH)

# === 2. Preprocessing ===
le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 3. Model Training ===

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
joblib.dump(rf, os.path.join(MODELS_PATH, 'random_forest_model.pkl'))

# SVM (RBF)
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
joblib.dump(svm, os.path.join(MODELS_PATH, 'svm_model.pkl'))

# KNN (best k selection)
best_k, best_acc = 3, 0
for k in range(3, 16, 2):
    knn_tmp = KNeighborsClassifier(n_neighbors=k)
    knn_tmp.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn_tmp.predict(X_test_scaled))
    if acc > best_acc:
        best_k, best_acc = k, acc

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
joblib.dump(knn, os.path.join(MODELS_PATH, 'knn_model.pkl'))

# === 4. Evaluation and Saving Results ===

def save_report_and_cm(model, model_name):
    y_pred = model.predict(X_test_scaled)
    # Classification report
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    with open(os.path.join(REPORTS_PATH, f'{model_name}_report.txt'), 'w') as f:
        f.write(report)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(CM_PATH, f'{model_name}_cm.png'))
    plt.close()
    # Return accuracy
    return accuracy_score(y_test, y_pred)

acc_rf = save_report_and_cm(rf, 'random_forest')
acc_svm = save_report_and_cm(svm, 'svm')
acc_knn = save_report_and_cm(knn, 'knn')

# === 5. Write Accuracy CSV ===
with open(ACCURACY_CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Accuracy'])
    writer.writerow(['Random Forest', round(acc_rf, 4)])
    writer.writerow(['SVM', round(acc_svm, 4)])
    writer.writerow(['KNN', round(acc_knn, 4)])

print('Done! All models, reports, confusion matrices, and accuracy CSV are saved in their folders.')
