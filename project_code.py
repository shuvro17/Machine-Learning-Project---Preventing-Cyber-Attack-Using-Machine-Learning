import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import glob
import matplotlib.pyplot as plt


csv_files = glob.glob("/kaggle/input/beth-dataset/*[!dns].csv")
data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)


print("Dataset shape:", data.shape)
print("\nFirst few rows:")
print(data.head())
print("\nMissing values:")
print(data.isnull().sum())


numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = data.select_dtypes(include=['object']).columns

print("\nNumeric columns:", numeric_columns.tolist())
print("Categorical columns:", categorical_columns.tolist())

# Handle missing values
if len(numeric_columns) > 0:
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
if len(categorical_columns) > 0:
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

# Encode categorical features
le_dict = {}
for col in categorical_columns:
    le_dict[col] = LabelEncoder()
    data[col] = le_dict[col].fit_transform(data[col])

# Feature-target split (using 'evil' as target)
X = data.drop(['evil', 'sus'], axis=1)  # Dropping both 'evil' and 'sus' columns
y = data['evil']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features (optional for Decision Tree but useful for consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree model
print("\nTraining Decision Tree model...")
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = dt_model.predict(X_test_scaled)

# Model evaluation
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=True, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

#feature selections
feature_importance.to_csv('feature_importances_decision_tree.csv', index=False)