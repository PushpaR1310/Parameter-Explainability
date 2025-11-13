import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

print("🔄 Loading kidney disease dataset...")
df = pd.read_csv('kidney_disease.csv')

print("📊 Dataset shape:", df.shape)
print("📋 Columns:", df.columns.tolist())

# Handle missing values
print("🔧 Handling missing values...")
df = df.dropna()

# Convert categorical variables to numerical
print("🔄 Converting categorical variables...")
categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

label_encoders = {}
for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Convert classification to binary (0 for notckd, 1 for ckd)
print("🔄 Converting target variable...")
df['classification'] = df['classification'].map({'notckd': 0, 'ckd': 1})

# Select features (same as in your app.py)
print("🎯 Selecting features...")
feature_columns = [
    'age', 'bp', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 
    'bgr', 'bu', 'sc', 'pot', 'wc', 'htn', 'dm', 'cad', 'pe', 'ane'
]

X = df[feature_columns]
y = df['classification']

print(f"📊 Features shape: {X.shape}")
print(f"📊 Target shape: {y.shape}")
print(f"📊 Target distribution: {y.value_counts()}")

# Split the data
print("🔄 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("🤖 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
print("📈 Evaluating model...")
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"✅ Training accuracy: {train_score:.4f}")
print(f"✅ Test accuracy: {test_score:.4f}")

# Save the model
print("💾 Saving model...")
with open('kidney_new.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model saved as 'kidney_new.pkl'")
print("🎉 Training completed successfully!")

# Test loading the model
print("🧪 Testing model loading...")
try:
    with open('kidney_new.pkl', 'rb') as file:
        test_model = pickle.load(file)
    print("✅ Model loads successfully!")
    print(f"✅ Model type: {type(test_model)}")
    print(f"✅ Model has predict method: {hasattr(test_model, 'predict')}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
