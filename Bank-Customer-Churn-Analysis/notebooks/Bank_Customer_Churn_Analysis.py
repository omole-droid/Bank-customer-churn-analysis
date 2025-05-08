
# Bank Customer Churn Prediction Project

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_excel('Cleaned_Bank_Customer_Data.xlsx')

# Encode categorical variables
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_country = LabelEncoder()
df['Geography'] = le_country.fit_transform(df['Geography'])

# Define features and target
X = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Gender', 'Geography']]
y = df['Exited']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict churn on test data
y_pred = model.predict(X_test)

# Calculate churn probability estimates
churn_probabilities = model.predict_proba(X)[:, 1]
df['Churn Probability Estimate'] = churn_probabilities

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC Score:", roc_auc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save final dataset with churn probability estimate
df.to_excel('Cleaned_Bank_Customer_Data_with_Churn_Probabilities.xlsx', index=False)
