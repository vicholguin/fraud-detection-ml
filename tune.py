from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

# Logistic Regression, Decision Trees, Random Forest, XGBoost, and Gradient Boosting Machines

# Load the dataset
file_path = "credit_card_fraud_dataset.csv"  # Update with correct file path
df = pd.read_csv(file_path)

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
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# adjust number of features 
n_features = X_train_scaled.shape[1]
n_features_to_select = min(10, n_features)

# Feature selection using Recursive Feature Elimination (RFE) with RandomForest
feature_selector = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(feature_selector, n_features_to_select=n_features_to_select)
rfe.fit(X_train_scaled, y_train_resampled)

# Select important features
selected_features = X_train.columns[rfe.support_]
X_train_selected = pd.DataFrame(X_train_scaled, columns=X_train.columns)[selected_features]
X_test_selected = pd.DataFrame(X_test_scaled, columns=X_test.columns)[selected_features]

# Decision Tree
dt_params = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10]
}

dt = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring="accuracy")
dt.fit(X_train, y_train)
print("Decision Tree: ", dt.best_params_, "->", accuracy_score(y_test, dt.predict(X_test)))

# Random Forest Parameters
rf_params = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2']
}

rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
rf.fit(X_train, y_train)
print("Random Forest:", rf.best_params_, "->", accuracy_score(y_test, rf.predict(X_test)))

# gradient boosting
gb_params = {
    "n_estimators": [50, 100],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5]
}
gb = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=5, scoring='accuracy')
gb.fit(X_train, y_train)
print("Gradient Boosting:", gb.best_params_, "->", accuracy_score(y_test, gb.predict(X_test)))

# XGBoost
xgb_params = {
    "n_estimators": [50, 100],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0]
}
xgb = GridSearchCV(XGBClassifier(eval_metric='mlogloss', random_state=42), xgb_params, cv=5, scoring='accuracy')
xgb.fit(X_train, y_train)
print("XGBoost:", xgb.best_params_, "->", accuracy_score(y_test, xgb.predict(X_test)))

# logistic regression
lr_params = {
    "penalty": ["l2"],
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs", "liblinear"],
    "max_iter": [100, 500, 1000]
}
lr = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=5, scoring='accuracy')
lr.fit(X_train, y_train)
print("Logistic Regression:", lr.best_params_, "->", accuracy_score(y_test, lr.predict(X_test)))
