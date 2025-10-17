import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# --- Configuration ---
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Helper Function for Saving ---
def save_model(model, filename):
    """Saves the trained model to a .pkl file."""
    filepath = os.path.join(FILE_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Successfully saved model to {filename}")
    except Exception as e:
        print(f"❌ Error saving model {filename}: {e}")

# =========================================================
# 1. CREATE AND SAVE CROP RECOMMENDATION MODEL (crop_model.pkl)
# =========================================================
print("\n--- 1. Creating crop_model.pkl (Recommendation) ---")

# A. Create Mock Data for Crop Recommendation (Classification)
np.random.seed(42) # for reproducible data

data_recommend = {
    'N': np.random.randint(40, 100, 100),
    'P': np.random.randint(10, 50, 100),
    'K': np.random.randint(10, 50, 100),
    'temperature': np.random.uniform(20, 35, 100),
    'humidity': np.random.uniform(50, 90, 100),
    'ph': np.random.uniform(5.5, 7.5, 100),
    'rainfall': np.random.uniform(100, 300, 100),
    'label': (['Rice'] * 30) + (['Maize'] * 30) + (['Wheat'] * 25) + (['Cotton'] * 15)
}
df_recommend = pd.DataFrame(data_recommend)

# B. Prepare Data
X_rec = df_recommend[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_rec = df_recommend['label']

# C. Train Model
crop_model = RandomForestClassifier(n_estimators=20, random_state=42)
crop_model.fit(X_rec, y_rec)

# D. Save Model
save_model(crop_model, 'crop_model.pkl')


# =========================================================
# 2. CREATE AND SAVE CROP PRICE PREDICTION MODEL (price_model.pkl)
# =========================================================
print("\n--- 2. Creating price_model.pkl (Price Prediction) ---")

# A. Create Mock Data for Price Prediction (Regression)
data_price = {
    # Categorical features
    'Crop_Name': np.tile(['Wheat', 'Rice', 'Maize', 'Cotton'], 25),
    'Market_Location': np.tile(['Market A', 'Market B'], 50),
    # Numerical features
    'Month': np.random.randint(1, 13, 100),
    # !!! UPDATED FEATURE NAME to align with the app's input intent !!!
    'Total_Production_kg': np.random.randint(50000, 300000, 100),
    'Historical_Rainfall_mm': np.random.uniform(100, 300, 100),
    'Previous_Year_Yield_Tons': np.random.randint(10000, 50000, 100),
}
df_price = pd.DataFrame(data_price)

# Introduce price variance (Base Price + Random Noise + Crop/Market Adjustment)
# NOTE: Price is inversely related to Production (higher production = lower price)
df_price['Price_per_Quintal'] = 2500 + np.random.uniform(-400, 400, 100)
df_price.loc[df_price['Crop_Name'] == 'Cotton', 'Price_per_Quintal'] += 4500
df_price.loc[df_price['Market_Location'] == 'Market B', 'Price_per_Quintal'] += 150
df_price['Price_per_Quintal'] -= (df_price['Total_Production_kg'] / 5000) # Price decreases with higher production

# B. Pre-process Data (Encode categorical features)
le = LabelEncoder()
df_price['Crop_Encoded'] = le.fit_transform(df_price['Crop_Name'])
df_price['Market_Encoded'] = le.fit_transform(df_price['Market_Location'])

# C. Prepare Data
# Features used for training. NOTE: The order is critical for app.py!
# We use Total_Production_kg directly here, removing the need for a proxy conversion.
X_price = df_price[['Crop_Encoded', 'Market_Encoded', 'Month', 'Total_Production_kg', 
                    'Historical_Rainfall_mm', 'Previous_Year_Yield_Tons']]
y_price = df_price['Price_per_Quintal']

# D. Train Model
price_model = RandomForestRegressor(n_estimators=20, random_state=42)
price_model.fit(X_price, y_price)

# E. Save Model
save_model(price_model, 'price_model.pkl')

print("\n--- Model Creation Complete. ---")
print("Make sure to run 'streamlit run app_streamlit.py' AFTER running this file.")