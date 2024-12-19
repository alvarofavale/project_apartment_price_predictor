import streamlit as st
import pandas as pd
import os
import numpy as np
import pickle
import plotly.express as px
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from streamlit_backend import load_csv, load_excel, prepare_data, load_model, make_predictions
from PIL import Image

# Define the paths to your files
data_file_path = r"C:\Users\faval\Desktop\Ironhack\DataAnalytics\final_project\data\clean\transformed_idealista_output_etl.csv"
model_file_path = r"C:\Users\faval\Desktop\Ironhack\DataAnalytics\final_project\moDels\rf_model.pkl"
province_map_path = r"C:\Users\faval\Desktop\Ironhack\DataAnalytics\final_project\data\clean\df_provinces_mapping_ipv.xls"
image = Image.open(r"C:\Users\faval\Desktop\Ironhack\DataAnalytics\final_project\images\real_estate_streamlit.jpg")
image2 = Image.open(r"C:\Users\faval\Desktop\Ironhack\DataAnalytics\final_project\images\real_estate_streamlit2.jpg")


# Load the csv
df_ml = load_csv(data_file_path)
# Load the excel
province_map = load_excel(province_map_path)

# Prepare the data
X_train, X_test, y_train, y_test = prepare_data(df_ml)

# Load the model
rf_model = load_model(model_file_path)

# Make predictions
y_pred_loaded = make_predictions(rf_model, X_test)

# Extract provinces
province_names = province_map['Province'].tolist()
province_codes = province_map['Code'].tolist()
province_dict = dict(zip(province_names, province_codes))

# User inputs
st.title("🏢 Apartment Price Prediction")

st.subheader("Get an estimate for your apartment based on your desired characteristics.")

# Display the first main image
st.image(image, caption='Welcome to the Apartment Price Predictor', use_column_width=True)

st.header("Enter Apartment Details Below:")


# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    selected_province = st.selectbox("Select the province:", options=province_names)
    features = {
        'ad_province': province_dict[selected_province],
        'ad_postalcode': st.number_input("Enter the postal code:", min_value=0),
        'ad_newconstruction': st.radio("Is it a new construction?", ["No", "Yes"]),
        'ad_area': st.number_input("Enter the area (in square meters):", min_value=0.0, step=1.0),
        'ad_roomnumber': st.number_input("Number of rooms:", min_value=0, step=1),
    }

with col2:
    features.update({
        'ad_hasswimmingpool': st.radio("Does it have a swimming pool?", ["No", "Yes"]),
        'ad_bathnumber': st.number_input("Number of bathrooms:", min_value=0, step=1),
        'ad_floornumber': st.number_input("Floor number:", min_value=0, step=1),
        'ad_haslift': st.radio("Does it have a lift?", ["No", "Yes"]),
        'ad_hasterrace': st.radio("Does it have a terrace?", ["No", "Yes"]),
        'ad_isintopfloor': st.radio("Is it on the top floor?", ["No", "Yes"]),
        'ad_hasgarden': st.radio("Does it have a garden?", ["No", "Yes"]),
        'ad_hasparkingspace': st.radio("Does it have a parking space?", ["No", "Yes"]),
    })

# Convert binary radio button answers to 0 or 1
binary_mappings = {"No": 0, "Yes": 1}
for key, value in features.items():
    if isinstance(value, str) and value in binary_mappings:
        features[key] = binary_mappings[value]

# Inverse Min-Max scaling for the transformed features
ad_area_min, ad_area_max = 15, 1485
ad_roomnumber_min, ad_roomnumber_max = 0, 99
ad_bathnumber_min, ad_bathnumber_max = 0, 401
ad_floornumber_min, ad_floornumber_max = 0, 60

features_adapted = {
    'ad_province': features['ad_province'],
    'ad_postalcode': features['ad_postalcode'], 
    'ad_newconstruction': features['ad_newconstruction'], 
    'ad_hasgarden': features['ad_hasgarden'],
    'ad_hasparkingspace': features['ad_hasparkingspace'], 
    'ad_hasswimmingpool': features['ad_hasswimmingpool'], 
    'ad_area': (features['ad_area'] - ad_area_min) / (ad_area_max - ad_area_min),
    'ad_roomnumber': (features['ad_roomnumber'] - ad_roomnumber_min) / (ad_roomnumber_max - ad_roomnumber_min),
    'ad_bathnumber': (features['ad_bathnumber'] - ad_bathnumber_min) / (ad_bathnumber_max - ad_bathnumber_min),
    'ad_floornumber': (features['ad_floornumber'] - ad_floornumber_min) / (ad_floornumber_max - ad_floornumber_min),
    'ad_haslift': features['ad_haslift'],  
    'ad_hasterrace': features['ad_hasterrace'], 
    'ad_isintopfloor': features['ad_isintopfloor']
}

# Create a DataFrame from the user input
input_data = pd.DataFrame([features_adapted])

st.image(image2, caption='Make your dreams come true', use_column_width=True)

# Make prediction
if st.button("Predict Price"):
    # Use the loaded model to predict the log price
    predicted_price_log = rf_model.predict(input_data)
    # Transform the predicted log price back to normal price
    predicted_price = np.exp(predicted_price_log) - 1  # Assuming log transformation was applied

    # Adjust price with IPV if available for the selected province
    province_code = features['ad_province']
    
    ipv_value = province_map.loc[province_map['Code'] == province_code, 'IPV'].values
    if ipv_value.size > 0:
        ipv = ipv_value[0]
        adjusted_price = predicted_price * (1 + ipv / 100)
        st.write(f"The adjusted predicted price of the apartment is: €{adjusted_price[0]:.2f}")
    else:
        st.write("No price index variation found for the provided province code.")
