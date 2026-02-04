"""
Script to train and save the Random Forest model for NYC Restaurant Health Inspection Prediction.
Run this script to generate the model file needed for the Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load cleaned data
print("Loading data...")
df = pd.read_csv('data/inspections_clean.csv')
print(f"Loaded {len(df)} records")

# Encode categorical features
print("Encoding categorical features...")
le_cuisine = LabelEncoder()
le_boro = LabelEncoder()

df['CUISINE_ENCODED'] = le_cuisine.fit_transform(df['CUISINE'].fillna('Unknown'))
df['BORO_ENCODED'] = le_boro.fit_transform(df['BORO'].fillna('Unknown'))

# Extract date features
df['INSPECTION_DATE'] = pd.to_datetime(df['INSPECTION_DATE'])
df['MONTH'] = df['INSPECTION_DATE'].dt.month
df['DAY_OF_WEEK'] = df['INSPECTION_DATE'].dt.dayofweek

# Select features
feature_cols = [
    'TOTAL_VIOLATIONS', 
    'CRITICAL_VIOLATIONS', 
    'MONTH', 
    'DAY_OF_WEEK',
    'CUISINE_ENCODED', 
    'BORO_ENCODED'
]

X = df[feature_cols]
y = df['GRADE']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nGrade distribution:")
print(y.value_counts())

# Train Random Forest model
print("\nTraining Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X, y)

print("Model trained successfully!")

# Save model
print("Saving model...")
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Also save encoders for use in the app
with open('models/cuisine_encoder.pkl', 'wb') as f:
    pickle.dump(le_cuisine, f)

with open('models/boro_encoder.pkl', 'wb') as f:
    pickle.dump(le_boro, f)

print("Model and encoders saved successfully!")
print(f"\nModel saved to: models/random_forest_model.pkl")
print(f"Feature importance:")
for i, col in enumerate(feature_cols):
    print(f"  {col}: {rf.feature_importances_[i]:.4f}")
