# ---------------------------------------------
# LUNG CANCER PREDICTION - FINAL PROJECT CODE
# ---------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------
# Load Dataset
# ---------------------------------------------
df = pd.read_csv("lung_cancer.csv")

# ---------------------------------------------
# Cleaning dataset
# ---------------------------------------------
df.drop_duplicates(inplace=True)

# Fixing inconsistent column names found in your report
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Map categorical values
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})

# ---------------------------------------------
# Feature Selection
# ---------------------------------------------
X = df[['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
         'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
         'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
         'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']]

y = df['LUNG_CANCER']

# ---------------------------------------------
# Train-Test Split
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# ---------------------------------------------
# Linear Regression (Not suitable, but included)
# ---------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
print("Linear Regression Accuracy:", lr.score(X_test, y_test) * 100)

# ---------------------------------------------
# Random Forest Classifier
# ---------------------------------------------
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
print("Random Forest Accuracy:", rfc.score(X_test, y_test) * 100)

# ---------------------------------------------
# K-Nearest Neighbors Classifier
# ---------------------------------------------
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("KNN Accuracy:", knn.score(X_test, y_test) * 100)

# ---------------------------------------------
# Decision Tree Classifier
# ---------------------------------------------
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
print("Decision Tree Accuracy:", dtc.score(X_test, y_test) * 100)

# ---------------------------------------------
# VISUALIZATION – ORIGINAL vs PREDICTED (Decision Tree)
# ---------------------------------------------
y_pred = dtc.predict(X_test)
df_plot = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})

df_plot_sorted = df_plot.sort_values(by='Actual')

plt.figure(figsize=(10, 6))
plt.plot(range(len(df_plot_sorted)), df_plot_sorted['Actual'], label='Original Class', marker='o')
plt.plot(range(len(df_plot_sorted)), df_plot_sorted['Predicted'], label='Predicted Class', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Class (0 = No Cancer, 1 = Cancer)')
plt.title('Original vs Predicted Class Labels')
plt.legend()
plt.show()

# ---------------------------------------------
# PIE CHART – LUNG CANCER DISTRIBUTION
# ---------------------------------------------
labels = ['No Cancer', 'Cancer']
sizes = df['LUNG_CANCER'].value_counts().sort_index()
explode = (0.1, 0)

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', explode=explode, shadow=True, startangle=90)
plt.title('Lung Cancer Distribution')
plt.show()

# ---------------------------------------------
# HEATMAP – CORRELATION MATRIX
# ---------------------------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
