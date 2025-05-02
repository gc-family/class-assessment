#!/usr/bin/env python
# coding: utf-8

# ========================================
# 1. Import Libraries
# ========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_theme(style='whitegrid')



# ========================================
# 2. Load Dataset
# ========================================
df = pd.read_csv('./dataset/student/student-por.csv', sep=';')
print("First 5 rows of the dataset:")
print(df.head())



df.shape


# ========================================
# 3. Exploratory Data Analysis (EDA)
# ========================================
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())





# Correlation matrix (numeric only)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()


# Histogram for numeric features
df.select_dtypes(include=['int64', 'float64']).hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.tight_layout()
plt.show()


# ========================================
# 4. Preprocessing
# ========================================
# Target variable
target = 'G3'

# Features
X = df.drop(columns=[target])
y = df[target]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing pipelines
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', sparse=False), categorical_cols)
])



# ========================================
# 5. Train/Test Split
# ========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# ========================================
# 6. Model Training
# ========================================
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)



# ========================================
# 7. Prediction & Evaluation
# ========================================
y_pred = pipeline.predict(X_test)

# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Residual plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual G3 Grade")
plt.ylabel("Predicted G3 Grade")
plt.title("Actual vs Predicted - Linear Regression")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.tight_layout()
plt.show()


# ========================================
# 8. Conclusion
# ========================================
print("\nConclusion:")
print("This model attempts to predict students' final grades (G3) using demographic and academic-related features.")
print("Linear Regression provides a baseline performance. You may improve results using more advanced models like Random Forest or Gradient Boosting.")
print("Further tuning and cross-validation can help in improving accuracy and generalization.")


get_ipython().system('jupyter nbconvert --to python student.ipynb --TemplateExporter.exclude_input_prompt=True')

