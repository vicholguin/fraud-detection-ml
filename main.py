# Group BETA
# Mario Rodriguez, Said Ajo-Montano, Emma Jackson, Victoria Holguin
# CS 487/519 - Applied Machine Learning I
# Group Project - ML approach to fraud detection

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, matthews_corrcoef
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

overall_start_time = time.time()

# Load datasets
df_synth = pd.read_csv("synthetic_fraud_dataset.csv")
df_credit = pd.read_csv("credit_card_fraud_dataset.csv")

# Standardize column names
df_synth.columns = df_synth.columns.str.strip()
df_credit.columns = df_credit.columns.str.strip()

# CREDIT CARD DATASET EDA
# 1. Class Distribution
plt.figure()
df_credit['IsFraud'].value_counts().plot(kind='bar', color=["blue", "red"])
plt.title("Credit Card Dataset - Class Distribution")
plt.xlabel("Class (0=Normal, 1=Fraud)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("credit_class_distribution.png")
plt.close()

# 2. Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df_credit.select_dtypes(include='number').corr(), cmap="coolwarm", center=0)
plt.title("Credit Card Dataset - Correlation Matrix")
plt.tight_layout()
plt.savefig("credit_correlation_matrix.png")
plt.close()

# 3. Transaction Amount Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df_credit['Amount'], bins=50, kde=True)
plt.title("Credit Card Dataset - Transaction Amount Distribution")
plt.xlabel("Amount")
plt.tight_layout()
plt.savefig("credit_transaction_amount.png")
plt.close()

# 4. Transactions by Day of Week
if 'TransactionDate' in df_credit.columns:
    df_credit['TransactionDate'] = pd.to_datetime(df_credit['TransactionDate'], errors='coerce')
    df_credit['DayOfWeek'] = df_credit['TransactionDate'].dt.dayofweek
    plt.figure(figsize=(10, 5))
    sns.countplot(x='DayOfWeek', hue='IsFraud', data=df_credit, palette='Set2')
    plt.title("Credit Card Dataset - Transactions by Day of Week")
    plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("credit_transactions_by_dayofweek.png")
    plt.close()

# SYNTHETIC DATASET
# 1. Class Distribution
plt.figure()
df_synth['Fraud_Label'].value_counts().plot(kind='bar', color=["blue", "red"])
plt.title("Synthetic Dataset - Class Distribution")
plt.xlabel("Class (0=Normal, 1=Fraud)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("synthetic_class_distribution.png")
plt.close()

# 2. Timestamp Parsing and Feature Engineering
if 'Timestamp' in df_synth.columns:
    df_synth['Timestamp'] = pd.to_datetime(df_synth['Timestamp'], errors='coerce')
    df_synth['Hour'] = df_synth['Timestamp'].dt.hour
    df_synth['DayOfWeek'] = df_synth['Timestamp'].dt.dayofweek

    plt.figure(figsize=(8, 5))
    sns.countplot(x='DayOfWeek', hue='Fraud_Label', data=df_synth, palette='Set2')
    plt.title("Synthetic Dataset - Transactions by Day of Week")
    plt.tight_layout()
    plt.savefig("synthetic_dayofweek_distribution.png")
    plt.close()

# 3. Top 3 Numeric Feature Histograms by Variance
numeric_cols = df_synth.select_dtypes(include='number').drop(columns=['Fraud_Label'], errors='ignore')
top_features = numeric_cols.var().sort_values(ascending=False).head(3).index

for col in top_features:
    normal_vals = pd.to_numeric(df_synth[df_synth['Fraud_Label'] == 0][col], errors='coerce').dropna()
    fraud_vals = pd.to_numeric(df_synth[df_synth['Fraud_Label'] == 1][col], errors='coerce').dropna()

    plt.figure()
    plt.hist(normal_vals, bins=50, alpha=0.5, label="Normal", density=True)
    plt.hist(fraud_vals, bins=50, alpha=0.5, label="Fraud", density=True)
    plt.title(f"Synthetic Dataset - {col} Histogram by Class")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"synthetic_hist_{col}.png")
    plt.close()

# Import SMOTE for handling class imbalance
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = "credit_card_fraud_dataset.csv"  # Update with correct file path
df = pd.read_csv(file_path)

# Use 10% of the data to help with performance
df = df.sample(frac=0.1, random_state=42)

# Standardize column names
df.rename(columns={
    "TransactionType": "Transaction_Type",
    "TransactionDate": "Timestamp", 
    "TransactionID": "Transaction_ID", 
    "IsFraud": "Fraud_Label"
}, inplace=True)

# Convert Timestamp to datetime and extract relevant time-based features
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Hour"] = df["Timestamp"].dt.hour
df["Day_of_Week"] = df["Timestamp"].dt.dayofweek
df["Is_Night"] = df["Hour"].apply(lambda x: 1 if x >= 20 or x < 6 else 0)

# Encode categorical features
categorical_features = ["Transaction_Type", "Device_Type", "Location", "Merchant_Category", "Card_Type", "Authentication_Method"]

# only include categorical columns in df
existing_categorical_features = [col for col in categorical_features if col in df.columns]

for col in existing_categorical_features:
    df[col] = LabelEncoder().fit_transform(df[col])

# Drop unnecessary columns
df.drop(["Transaction_ID", "Timestamp"], axis=1, inplace=True)

if "User_ID" in df.columns:
    df.drop(["User_ID"], axis=1, inplace=True)

if "MerchantID" in df.columns:
    df.drop(["MerchantID"], axis=1, inplace=True)

# Define features and target
X = df.drop("Fraud_Label", axis=1)
y = df["Fraud_Label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance fraud vs non-fraud cases
# Reduce oversampling to help with performance
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# adjust number of features 
n_features = X_train_scaled.shape[1]
# Select fewer features to help with performance
# n_features_to_select = min(10, n_features)
n_features_to_select = 5

# Feature selection using Recursive Feature Elimination (RFE) with RandomForest
# Lower number of estimators to help with performance
# feature_selector = RandomForestClassifier(n_estimators=50, random_state=42)
feature_selector = RandomForestClassifier(n_estimators=10, random_state=42)
rfe = RFE(feature_selector, n_features_to_select=n_features_to_select)
rfe.fit(X_train_scaled, y_train_resampled)

# Select important features
selected_features = X_train.columns[rfe.support_]
X_train_selected = pd.DataFrame(X_train_scaled, columns=X_train.columns)[selected_features]
X_test_selected = pd.DataFrame(X_test_scaled, columns=X_test.columns)[selected_features]

# Define models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='newton-cg'),
    # Lower number of estimators to help with performance
    "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
    # Lower number of estimators to help with performance
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=10, random_state=42),
    "Support Vector Machine": SVC(kernel='linear', probability=True),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42)
}

# Parameter grid for hyperparameter tuning
# Define a smaller grid to help with performance
param_grid = {
    'n_estimators': [10, 50],
    'max_depth': [3, 5]
}

# Store results
results = []

for model_name, model in models.items():
    start_time = time.time()
    model.fit(X_train_selected, y_train_resampled)
    
    # Get probability predictions and adjust threshold to detect fraud
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    y_pred = (y_pred_proba > 0.3).astype(int)  # Lower threshold to detect more fraud cases

    train_time = time.time() - start_time

    # Evaluate Model
    roc_auc = roc_auc_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=1)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({"Model": model_name, "ROC-AUC": roc_auc, "Precision": precision, "Recall": recall, "F1-Score": f1, "MCC": mcc, "Time": train_time})

# Convert to DataFrame and save results
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison_results_fixed3.csv", index=False)

# Print results
print(results_df)

# Visualisation for model performance
# Plotting the ROC Curve
from sklearn.metrics import roc_curve

plt.figure()
for model_name, model in models.items():
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=model_name)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.close()

# Visualisation for feature importance
# Access the fitted estimator inside RFE to get feature importances
importances = rfe.estimator_.feature_importances_

# Plotting feature importance
plt.barh(selected_features, importances)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

total_time = time.time() - overall_start_time
print(f"Overall run time: {total_time:.4f} seconds")
