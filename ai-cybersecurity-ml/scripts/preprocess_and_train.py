import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Set paths
DATA_PATH = "datasets/KDDTrain+.txt"
MODEL_PATH = "models/rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# NSL-KDD column names (43 columns including difficulty_level)
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

print("🔽 Loading dataset...")
df = pd.read_csv(DATA_PATH, names=columns)

# Drop difficulty_level (not needed for training)
df.drop('difficulty_level', axis=1, inplace=True)

# Binary classification: normal vs attack
df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

# Encode categorical features
cat_cols = ['protocol_type', 'service', 'flag']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Split features and labels
X = df.drop('label', axis=1)
y = df['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
print("🧠 Training model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
print("✅ Model Evaluation:\n")
print(classification_report(y_test, y_pred))

# Save model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(clf, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\n💾 Model and scaler saved to 'models/' directory.")
