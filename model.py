# =========================================
# IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")


# =========================================
# LOAD DATASET
# =========================================
df = pd.read_csv("Dataset/student_risk_dataset_5000.csv")

X = df.drop("Dropout_Risk", axis=1)
y = df["Dropout_Risk"]


# =========================================
# SCALING & SPLIT
# =========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# =========================================
# TRAIN MODEL (XGBOOST)
# =========================================
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")
print("Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))


# =========================================
# TAKE USER INPUT
# =========================================
print("\n--- ENTER STUDENT DETAILS ---")

attendance = float(input("Enter Attendance Percentage (0–100): "))
units = int(input("Enter Number of Enrolled Units: "))
grade = float(input("Enter Average Grade (0–10): "))
engagement = int(input("Enter Engagement Score (1–10): "))
task_completion = float(input("Enter Task Completion Rate (0–100): "))


# =========================================
# PREDICTION
# =========================================
student_data = np.array([[attendance, units, grade, engagement, task_completion]])
student_scaled = scaler.transform(student_data)

prediction = model.predict(student_scaled)[0]
probability = model.predict_proba(student_scaled)[0][1]


# =========================================
# OUTPUT
# =========================================
print("\n--- PREDICTION RESULT ---")

if prediction == 1:
    print("Dropout Risk: HIGH RISK")
else:
    print("Dropout Risk: LOW RISK")

print("Risk Probability:", round(probability, 2))
