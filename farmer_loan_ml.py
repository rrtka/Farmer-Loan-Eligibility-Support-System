from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("farmer_loan_dataset.csv")

# Encode categorical columns
cat_cols = ["education","region","crop_type","irrigation","past_loan_status","loan_purpose","risk_label"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("eligible", axis=1)
y = df["eligible"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data Preprocessing Done!")
print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

print("Models trained successfully!")
models = {"Logistic Regression": log_model, "Random Forest": rf_model, "Decision Tree": dt_model}

for name, model in models.items():
    preds = model.predict(X_test)
    print(f"\n{name}:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
import matplotlib.pyplot as plt
import seaborn as sns

# Feature importance from Random Forest
importances = rf_model.feature_importances_
features = X.columns

# Create a DataFrame
feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(12,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title("Feature Importance (Random Forest)")
plt.show()
