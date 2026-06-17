import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# Load Dataset
# =========================
csv_file = "student-mat.csv"  

df = pd.read_csv(csv_file, sep=';')

print("\nDataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# Data Visualization
# =========================

# Histogram of Final Grades
plt.figure(figsize=(8, 5))
plt.hist(df['G3'], edgecolor='black')
plt.title('Final Score Distribution')
plt.xlabel('G3 Score')
plt.ylabel('Count')
plt.show()

# Grade Counts
plt.figure(figsize=(10, 8))
df['G3'].value_counts().sort_values().plot.barh(
    color=sns.color_palette('inferno', 40)
)
plt.title('Number of Students Who Scored a Particular Grade')
plt.xlabel('Number of Students')
plt.ylabel('Final Grade')
plt.show()

# =======================
# Age Distribution
# =======================

plt.figure(figsize=(8, 5))
sns.kdeplot(df['age'], fill=True)
plt.title('Ages of Students')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()

# =========================
# Data Preprocessing
# =========================

label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# =========================
# Features and Target
# =========================

X = df.drop('G3', axis=1)
y = df['G3']

# =========================
# Train Test Split
# =========================

xtrain, xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Linear Regression Model
# =========================

model = LinearRegression()
model.fit(xtrain, ytrain)

# =========================
# Prediction
# =========================

y_pred = model.predict(xtest)

# =========================
# Evaluation
# =========================

mse = mean_squared_error(ytest, y_pred)
r2 = r2_score(ytest, y_pred)

print("\nModel Evaluation")
print("-" * 30)
print(f"Mean Squared Error : {mse:.4f}")
print(f"R² Score           : {r2:.4f}")

results = pd.DataFrame({
    "Actual": ytest.values,
    "Predicted": np.round(y_pred, 2)
})

print("\nFirst 10 Predictions:")
print(results.head(10))
