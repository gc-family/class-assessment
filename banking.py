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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier  # Chosen algorithm

sns.set_theme(style='whitegrid')


# ========================================
# 2. Load Dataset
# ========================================
df = pd.read_csv('./dataset/bank/bank.csv')  # Adjust path if needed
df.head()


df.shape


df['deposit'].value_counts()


df.info()


# ========================================
# 3. Exploratory Data Analysis (EDA)
# ========================================
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Visualize target distribution
sns.countplot(x='deposit', data=df)
plt.show()

# Visualize balance distribution
sns.histplot(df['balance'], bins=50, kde=True)
plt.show()



# ========================================
# 4. Preprocessing
# ========================================

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Handle missing values if any (none expected)
# df.fillna(method='ffill', inplace=True)

# Define features and target
X = df.drop('deposit', axis=1)
y = df['deposit']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# ========================================
# 5. Train/Test Split
# ========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# ========================================
# 6. Model Training
# ========================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# ========================================
# 7. Prediction & Evaluation
# ========================================
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print("classification_report")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))



# ========================================
# 8. Feature Importance (Optional)
# ========================================
importances = model.feature_importances_
feature_names = df.drop('deposit', axis=1).columns

feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(12,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title('Feature Importance')
plt.show()

# ========================================
# 9. Conclusion
# ========================================
# You can summarize performance here manually or print observations


get_ipython().system('jupyter nbconvert --to python banking.ipynb --TemplateExporter.exclude_input_prompt=True')

