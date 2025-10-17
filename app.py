import streamlit as st
import numpy as np
import pickle
import pandas as pd
import json
import hashlib
import os
from sklearn.preprocessing import LabelEncoder 
from datetime import datetime

# --- 0. Setup and Constants ---

USERS_FILE = 'users.json'
# Using a simple JSON file to store listed equipment for the prototype
LISTINGS_FILE = 'equipment_listings.json' 

# --- Streamlit Session State Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'booking_equipment' not in st.session_state:
    st.session_state['booking_equipment'] = None

# --- 1. Model Loading and Mappings ---

@st.cache_resource
def load_models_and_mappings():
    """Loads models and defines constants needed for the app."""
    CROP_MAPPING = {'Maize': 0, 'Rice': 1, 'Wheat': 2, 'Cotton': 3} 
    MARKET_MAPPING = {'Market A': 0, 'Market B': 1}
    MOCK_HISTORICAL_DATA = {
        'Historical_Rainfall_mm': 150.0,
        'Previous_Year_Yield_Tons': 30000.0,
        'Average_Yield_kg_per_Hectare': 3000.0
    }
    
    crop_model = None
    price_model = None
    
    try:
        with open('crop_model.pkl', 'rb') as f: crop_model = pickle.load(f)
    except FileNotFoundError: pass
    
    try:
        with open('price_model.pkl', 'rb') as f: price_model = pickle.load(f)
    except FileNotFoundError: pass

    return crop_model, price_model, CROP_MAPPING, MARKET_MAPPING, MOCK_HISTORICAL_DATA

crop_model, price_model, CROP_MAPPING, MARKET_MAPPING, MOCK_HISTORICAL_DATA = load_models_and_mappings()


# --- 2. Authentication and Listings Functions ---

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    try:
        with open(USERS_FILE, 'r') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f: json.dump(users, f, indent=4)

def register_user(username, password):
    users = load_users()
    if username in users:
        st.error("Username already exists. Please choose a different one.")
        return False
    if len(password) < 6:
        st.error("Password must be at least 6 characters long.")
        return False
    users[username] = {'password_hash': hash_password(password)}
    save_users(users)
    st.success("Registration successful! Please log in.")
    return True

def login_user(username, password):
    users = load_users()
    if username in users and users[username]['password_hash'] == hash_password(password):
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.success(f"Welcome, {username}! Accessing SmartFarm.")
        st.rerun()
        return True
    else:
        st.error("Invalid Username or Password.")
        return False

def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = ''
    st.session_state['booking_equipment'] = None
    st.rerun()

# NEW Listing Functions
def load_listings():
    """Loads equipment listings from the JSON file."""
    try:
        with open(LISTINGS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_listings(listings):
    """Saves equipment listings to the JSON file."""
    with open(LISTINGS_FILE, 'w') as f:
        json.dump(listings, f, indent=4)


# --- 3. Prediction and Data Functions (Unchanged logic) ---

def get_crop_description(crop_name):
    descriptions = {
        "Rice": {"description": "Rice is a staple crop that thrives in regions with high rainfall and high temperatures.", "reasoning": "The AI recommends Rice due to the **high Nitrogen (N) and high Rainfall** inputs, indicating conditions suitable for flooded or semi-aquatic cultivation."},
        "Maize": {"description": "Maize (Corn) is a versatile crop that requires moderate rainfall and warm temperatures, but can tolerate a wide range of soil conditions.", "reasoning": "The AI recommends Maize because the nutrient levels, particularly **low N and P**, suggest less fertile land, which Maize can adapt to well, especially with moderate rainfall."},
        "Wheat": {"description": "Wheat is a major cereal crop typically grown in cooler climates with moderate to high potassium (K) levels.", "reasoning": "The AI suggests Wheat because the **temperature is moderate to low** and the **pH level is near neutral**, creating ideal conditions for winter crop cultivation."},
        "Cotton": {"description": "Cotton is a commercial fiber crop that needs a warm, long growing season with plenty of sunlight and well-drained, deep soil.", "reasoning": "The AI suggests Cotton because the **temperature is consistently high** and the **humidity and rainfall are low**, aligning with the dry, sunny conditions cotton prefers during its ripening stage."},
        "Fallback Crop": {"description": "This is a basic crop recommendation based on limited data (fallback mode).", "reasoning": "The AI is running on fallback logic because the trained model file (`crop_model.pkl`) could not be loaded. Please train and save the model for accurate results."}
    }
    clean_name = crop_name.split(' (Fallback)')[0]
    return descriptions.get(clean_name, descriptions["Fallback Crop"])

def predict_crop(features):
    if crop_model:
        prediction = crop_model.predict(features)
        return prediction[0]
    else:
        N, P, _, _, _, _, _ = features[0]
        if N > 90 and P > 40: return "Rice (Fallback)"
        elif N < 50 and P < 30: return "Maize (Fallback)"
        else: return "Wheat (Fallback)"

def predict_price(crop_name, market_location, month, weight_kg):
    crop_encoded = CROP_MAPPING.get(crop_name)
    market_encoded = MARKET_MAPPING.get(market_location)
    
    if crop_encoded is None or market_encoded is None: return "Error: Invalid selection for encoding."
    
    features = np.array([[
        crop_encoded, market_encoded, month, weight_kg, 
        MOCK_HISTORICAL_DATA['Historical_Rainfall_mm'], MOCK_HISTORICAL_DATA['Previous_Year_Yield_Tons']
    ]])
    
    predicted_price_per_quintal = 0.0

    if price_model: predicted_price_per_quintal = price_model.predict(features)[0]
    else:
        base_price = 2000 + (market_encoded * 50) - (weight_kg / 10000)
        if crop_name == 'Cotton': base_price = max(6000, base_price)
        predicted_price_per_quintal = base_price
            
    return round(float(predicted_price_per_quintal), 2)


# --- Data Generation Functions for Technical Deep Dive ---
def generate_mock_crop_data():
    np.random.seed(42)
    data = {
        'N': np.random.randint(40, 100, 10), 'P': np.random.randint(10, 50, 10), 'K': np.random.randint(10, 50, 10), 
        'temperature': np.random.uniform(20, 35, 10).round(2), 'humidity': np.random.uniform(50, 90, 10).round(2), 
        'ph': np.random.uniform(5.5, 7.5, 10).round(2), 'rainfall': np.random.uniform(100, 300, 10).round(2), 
        'label': np.tile(['Rice', 'Maize', 'Wheat', 'Cotton'], 3)[:10]
    }
    return pd.DataFrame(data)

def generate_mock_price_data():
    np.random.seed(42)
    data_price = {
        'Crop_Name': np.tile(['Wheat', 'Rice'], 5), 'Market_Location': np.tile(['Market A', 'Market B'], 5), 
        'Month': np.random.randint(1, 13, 10), 'Total_Production_kg': np.random.randint(50000, 300000, 10), 
        'Price_per_Quintal': np.random.uniform(1800, 3000, 10).round(2)
    }
    df_price = pd.DataFrame(data_price)
    df_price.loc[df_price['Crop_Name'] == 'Wheat', 'Price_per_Quintal'] += 500 
    return df_price


# --- 4. Main App Structure and UI ---

st.set_page_config(
    page_title="SmartFarm Assist Prototype",
    layout="wide",
    initial_sidebar_state="auto"
)

# ----------------------------------------------------
# A. Login/Registration Page
# ----------------------------------------------------
if not st.session_state['logged_in']:
    st.title("🌾 SmartFarm Assist — Login / Register")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Farmer Login")
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_user")
            login_password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Log In")
            if submitted: login_user(login_username, login_password)

    with tab2:
        st.subheader("New Farmer Registration")
        with st.form("register_form"):
            reg_username = st.text_input("Choose Username", key="reg_user")
            reg_password = st.text_input("Choose Password (min 6 chars)", type="password", key="reg_pass")
            submitted = st.form_submit_button("Register Account")
            if submitted: register_user(reg_username, reg_password)


# ----------------------------------------------------
# B. Main Application
# ----------------------------------------------------
else:
    # Sidebar components for logged-in state
    st.sidebar.title(f"Welcome, {st.session_state['username']}!")
    if st.sidebar.button("Logout"): logout()

    st.title("🌾 SmartFarm Assist — Prototype")
    
    feature = st.sidebar.radio(
        "Select Feature",
        ("Crop Recommendation", "Crop Price Prediction", "Equipment Rental", "Technical Deep Dive")
    )
    
    # --- Crop Recommendation ---
    if feature == "Crop Recommendation":
        st.header("🌱 Crop Recommendation")
        with st.form("recommendation_form"):
            col1, col2, col3 = st.columns(3)
            N = col1.number_input("Nitrogen (N)", value=90.00, step=1.0)
            P = col2.number_input("Phosphorus (P)", value=40.00, step=1.0)
            K = col3.number_input("Potassium (K)", value=40.00, step=1.0)
            temp = col1.number_input("Temperature (°C)", value=25.00, step=0.1, key='temp_rec')
            humidity = col2.number_input("Humidity (%)", value=80.00, step=0.1, key='humid_rec')
            ph = col3.number_input("pH", value=6.50, step=0.01, key='ph_rec')
            rainfall = st.number_input("Rainfall (mm)", value=200.00, step=1.0)
            submitted = st.form_submit_button("✨ Recommend Crop")
            
            if submitted:
                features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
                recommended_crop = predict_crop(features)
                explanation = get_crop_description(recommended_crop)
                st.success(f"**Recommended Crop:** {recommended_crop}")
                st.markdown("---"); st.markdown(f"**Description:** {explanation['description']}")
                st.markdown(f"**AI Rationale:** {explanation['reasoning']}")

    # --- Crop Price Prediction ---
    elif feature == "Crop Price Prediction":
        st.header("💰 Crop Price Prediction")
        with st.form("price_prediction_form"):
            col1, col2 = st.columns(2)
            crop_name = col1.selectbox("Crop Name", options=list(CROP_MAPPING.keys()))
            market_location = col2.selectbox("Market Location", options=list(MARKET_MAPPING.keys()))
            col3, col4 = st.columns(2)
            month = col3.number_input("Month of Sale (1-12)", value=10, min_value=1, max_value=12, step=1)
            weight_kg = col4.number_input("Weight (kg)", value=50000, min_value=100)

            submitted = st.form_submit_button("🔮 Predict Price")
            
            if submitted:
                predicted_price = predict_price(crop_name, market_location, month, weight_kg)
                if isinstance(predicted_price, str):
                    st.error(predicted_price)
                else:
                    st.success(f"**Predicted Price (₹/Quintal):** ₹{predicted_price:,.2f}")
                    st.info("The prediction is the expected price per quintal (100 kg).")

    # --- Equipment Rental (UPDATED with Tabs) ---
    elif feature == "Equipment Rental":
        st.header("🚜 Equipment Rental Marketplace")
        
        tab_borrow, tab_lend, tab_view = st.tabs(["Borrow Equipment", "Rent Out Your Equipment", "Active Listings"])

        # --- TAB 1: Borrow Equipment (Booking) ---
        with tab_borrow:
            st.subheader("1. Browse and Book")
            
            # --- Mock equipment list (static) ---
            st.markdown("**Available Equipment (Demo Pricing):**")
            col_list, col_book = st.columns(2)
            
            with col_list:
                st.markdown("* **Tractor:** ₹500/day")
                st.markdown("* **Harvester:** ₹800/day")
            with col_book:
                if st.button("Book Tractor", key='book_tractor_btn'): st.session_state['booking_equipment'] = 'Tractor'
                if st.button("Book Harvester", key='book_harvester_btn'): st.session_state['booking_equipment'] = 'Harvester'

            col_list2, col_book2 = st.columns(2)
            with col_list2:
                st.markdown("* **Seeder:** ₹300/day")
                st.markdown("* **Rotavator:** ₹350/day")
            with col_book2:
                if st.button("Book Seeder", key='book_seeder_btn'): st.session_state['booking_equipment'] = 'Seeder'
                if st.button("Book Rotavator", key='book_rotavator_btn'): st.session_state['booking_equipment'] = 'Rotavator'


            # --- Booking Form (Dynamically appears) ---
            st.markdown("---")
            if 'booking_equipment' in st.session_state and st.session_state['booking_equipment']:
                equipment = st.session_state['booking_equipment']
                st.info(f"You are booking: **{equipment}**")
                
                with st.form(key='details_form'):
                    st.markdown("### 📝 Enter Rental Details")
                    name = st.text_input("Full Name", value=st.session_state['username'].replace('_', ' ').title())
                    contact = st.text_input("Contact Number")
                    address = st.text_area("Delivery Address (Full Address, Pincode)")
                    
                    col_start, col_end = st.columns(2)
                    start_date = col_start.date_input("Start Date", min_value=datetime.today())
                    end_date = col_end.date_input("End Date", min_value=datetime.today())
                    
                    final_submit = st.form_submit_button(f"Confirm Booking for {equipment}")
                    
                    if final_submit:
                        if not all([name, contact, address]): st.error("Please fill in all required fields.")
                        elif start_date >= end_date: st.error("End Date must be after the Start Date.")
                        else:
                            st.success(f"✅ Booking Confirmed for **{equipment}**!")
                            st.balloons()
                            st.session_state['booking_equipment'] = None

        # --- TAB 2: Rent Out Your Equipment (Listing) ---
        with tab_lend:
            st.subheader("2. List Your Equipment for Rent")
            st.warning("By listing, you agree to connect with other farmers for rental arrangements.")

            with st.form(key='listing_form'):
                lender_name = st.text_input("Your Full Name", value=st.session_state['username'].replace('_', ' ').title())
                lender_contact = st.text_input("Your Contact Number", key='lender_contact')
                
                equipment_type = st.selectbox("Equipment Type", ['Tractor', 'Harvester', 'Seeder', 'Rotavator', 'Other'])
                model_year = st.number_input("Model Year", min_value=1950, max_value=datetime.now().year, value=2015)
                price_per_day = st.number_input("Rental Price (₹) per Day", min_value=100, step=50)
                equipment_location = st.text_area("Equipment Location (Village/City, Pincode)")
                
                list_submit = st.form_submit_button("✅ Submit Equipment Listing")

                if list_submit:
                    if not all([lender_name, lender_contact, equipment_location]):
                        st.error("Please fill in all required fields.")
                    else:
                        listings = load_listings()
                        new_listing = {
                            'id': len(listings) + 1,
                            'lender': st.session_state['username'],
                            'type': equipment_type,
                            'model_year': model_year,
                            'price_per_day': price_per_day,
                            'location': equipment_location,
                            'contact': lender_contact,
                            'date_listed': datetime.now().strftime("%Y-%m-%d")
                        }
                        listings.append(new_listing)
                        save_listings(listings)
                        st.success(f"🎉 Your **{equipment_type}** has been successfully listed!")
                        st.snow()

        # --- TAB 3: Active Listings ---
        with tab_view:
            st.subheader("3. All Active Listings")
            listings = load_listings()
            if listings:
                df_listings = pd.DataFrame(listings)
                df_display = df_listings[['type', 'model_year', 'price_per_day', 'location', 'contact', 'date_listed']]
                df_display.columns = ['Equipment Type', 'Model Year', 'Price (₹)/Day', 'Location', 'Contact No.', 'Date Listed']
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                st.markdown(f"*{len(listings)} active listings found.*")
            else:
                st.info("No equipment has been listed yet. Be the first!")


    # --- Technical Deep Dive ---
    elif feature == "Technical Deep Dive":
        st.header("🔬 Technical Deep Dive: ML Implementation")
        
        st.subheader("1. Crop Recommendation Model")
        st.markdown(f"**Model:** **RandomForestClassifier** (Saved as `crop_model.pkl`).")
        if crop_model: st.success("Model loaded successfully.")
        else: st.error("Model file not found. Fallback logic is active.")

        st.markdown("#### Sample Dataset (10 Rows)"); df_crop_sample = generate_mock_crop_data()
        st.dataframe(df_crop_sample)
        
        st.subheader("2. Crop Price Prediction Model")
        st.markdown(f"**Model:** **RandomForestRegressor** (Saved as `price_model.pkl`).")
        st.markdown("""
        The model uses **Total Production (kg)**, Market, and Time of Sale to predict the Price per Quintal (100 kg), reflecting real-world supply-demand economics.
        """)
        if price_model: st.success("Model loaded successfully.")
        else: st.error("Model file not found. Fallback logic is active.")
            
        st.markdown("#### Sample Dataset (10 Rows - Features used in Training)"); df_price_sample = generate_mock_price_data()
        st.dataframe(df_price_sample)