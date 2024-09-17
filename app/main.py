import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Get clean data
def clean_data():
    data = pd.read_csv('data/catalog.csv')
    
    # Drop unnecessary columns
    data = data.drop(['continent_code', 'country_code', 'country_name', 'state/province', 
                      'population', 'city/town', 'distance', 'location_description', 
                      'latitude', 'longitude', 'geolocation', 'source_name', 'source_link'], axis=1)
    
    # Drop rows where 'trigger' is NaN
    data = data.dropna(subset=['trigger'])
    
    # Map the 'trigger' column to numerical values
    data['trigger'] = data['trigger'].map({
        'Rain': 1,
        'Downpour': 2,
        'Unknown': 3,
        'Tropical cyclone': 4,
        'Flooding': 5,
        'Continous rain': 6
    })
    
    # Drop rows with NaN values in 'trigger' after mapping (if any)
    data = data.dropna(subset=['trigger'])
    
    return data

# Create model
def create_model(data):
    # Separate features (X) and target (y)
    X = data.drop(['trigger'], axis=1)
    y = data['trigger']
    
    # One-Hot Encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Handle missing values in X by imputing with the mean
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Scale the numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_prediction = model.predict(X_test)
    
    print('Accuracy of model:', accuracy_score(y_test, y_prediction))
    print('Classification Report:\n', classification_report(y_test, y_prediction))

    return model, scaler

# Define sidebar for user input
def define_sidebar():
    st.sidebar.header("Landslide Factors")

    data = clean_data()
    
    # Define the slider and selectbox labels
    slider_labels = [
        ("Trigger", "trigger"),
        ("Hazard Type", "hazard_type"),
        ("Landslide Type", "landslide_type"),
        ("Landslide Size", "landslide_size"),
        ("Storm Name", "storm_name"),
        ("Injuries", "injuries"),
        ("Fatalities", "fatalities"),
    ]

    input_dict = {}
    for label, key in slider_labels:
        if key in data.columns:
            # Handle numerical columns with sliders
            if pd.api.types.is_numeric_dtype(data[key]):
                input_dict[key] = st.sidebar.slider(
                    label,
                    min_value=float(data[key].min()),
                    max_value=float(data[key].max()),
                    value=float(data[key].mean())
                )
            else:
                # Handle categorical columns with selectbox
                unique_values = data[key].dropna().unique()
                
                # Ensure unique values are sorted correctly
                if all(isinstance(val, (str, np.str_)) for val in unique_values):
                    unique_values = sorted(set(unique_values))
                else:
                    unique_values = sorted(unique_values, key=lambda x: (isinstance(x, str), x))
                
                selected_value = st.sidebar.selectbox(
                    label,
                    options=unique_values,
                    index=list(unique_values).index(data[key].mode()[0]) if data[key].mode().size > 0 else 0
                )
                input_dict[key] = selected_value
        else:
            st.warning(f"Warning: {key} not found in data columns.")
    
    return input_dict

# Scale the input values
def scaled_values(input_dict):
    data = clean_data()
    
    # Drop 'trigger' column to work with the rest of the features
    X = data.drop(['trigger'], axis=1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        if key in X.columns:
            if pd.api.types.is_numeric_dtype(X[key]):
                # Numeric column: scale the value
                max_val = X[key].max()
                min_val = X[key].min()
                # Ensure max_val is not equal to min_val to avoid division by zero
                if max_val != min_val:
                    scaled_value = (value - min_val) / (max_val - min_val)
                else:
                    scaled_value = 0  # or handle appropriately if max and min are equal
                scaled_dict[key] = scaled_value
            else:
                # Categorical column: handle scaling differently
                unique_values = X[key].dropna().unique()
                if len(unique_values) > 1:
                    # If there is more than one unique value, scale accordingly
                    if all(isinstance(val, (int, float)) for val in unique_values):
                        # If categorical values are numeric, scale as numeric
                        max_val = max(unique_values)
                        min_val = min(unique_values)
                        scaled_value = (value - min_val) / (max_val - min_val)
                    else:
                        # For non-numeric categories, scale by index
                        unique_values = sorted(set(unique_values))
                        scaled_value = unique_values.index(value) / (len(unique_values) - 1)
                else:
                    # Handle case where there is only one unique value
                    scaled_value = 0  # or handle appropriately if only one unique value exists
                scaled_dict[key] = scaled_value
        else:
            st.warning(f"Warning: {key} not found in data columns.")
    
    return scaled_dict


# Create radar chart
def clean_data_chart(input_data):
    input_data = scaled_values(input_data)

    categories = ['Radius (mean)', 'Texture (mean)', 'Perimeter (mean)', 'Area (mean)', 
                'Smoothness (mean)', 'Compactness (mean)', 
                'Concavity (mean)', 'Concave points (mean)',
                'Symmetry (mean)', 'Fractal dimension (mean)']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
            r=[
            input_data.get('radius_mean', 0), input_data.get('texture_mean', 0), input_data.get('perimeter_mean', 0),
            input_data.get('area_mean', 0), input_data.get('smoothness_mean', 0), input_data.get('compactness_mean', 0),
            input_data.get('concavity_mean', 0), input_data.get('concave points_mean', 0), input_data.get('symmetry_mean', 0),
            input_data.get('fractal_dimension_mean', 0)
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data.get('radius_se', 0), input_data.get('texture_se', 0), input_data.get('perimeter_se', 0),
            input_data.get('area_se', 0), input_data.get('smoothness_se', 0), input_data.get('compactness_se', 0),
            input_data.get('concavity_se', 0), input_data.get('concave points_se', 0), input_data.get('symmetry_se', 0),
            input_data.get('fractal_dimension_se', 0)
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data.get('radius_worst', 0), input_data.get('texture_worst', 0), input_data.get('perimeter_worst', 0),
            input_data.get('area_worst', 0), input_data.get('smoothness_worst', 0), input_data.get('compactness_worst', 0),
            input_data.get('concavity_worst', 0), input_data.get('concave points_worst', 0), input_data.get('symmetry_worst', 0),
            input_data.get('fractal_dimension_worst', 0)
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    
    return fig

# Add predictions to the sidebar
def add_predictions(input_data):
    # Load model and scaler
    model = pickle.load(open("models/model.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    
    # Convert input_dict to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply the same preprocessing steps as during model training
    # One-Hot Encode categorical features
    input_df = pd.get_dummies(input_df, drop_first=True)
    
    # Ensure that the DataFrame has the same columns as the training data
    X_train = pd.read_csv('data/catalog.csv').drop(['trigger'], axis=1)
    X_train = pd.get_dummies(X_train, drop_first=True)
    missing_cols = set(X_train.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[X_train.columns]
    
    # Scale the input values
    input_array = np.array(input_df).astype(float)
    input_array_scaled = scaler.transform(input_array)
    
    # Make the prediction
    prediction = model.predict(input_array_scaled)
    
    # Display prediction and probabilities
    st.subheader("Landslide Prediction")
    st.write("The landslide prediction is:")
    
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
        
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
    
    st.write("This app can assist in predicting landslides but should not be used as a sole decision-making tool.")

# Main function to configure the app
def main():
    st.set_page_config(
        page_title="Landslide Predictor",
        page_icon="🌋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Configure CSS
    with open('assets/styles.css') as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    # Create the sidebar and get input data
    input_data = define_sidebar()

    # Create headers and layout
    with st.container():
        st.title("Landslide Predictor")
        st.write("The Landslide Predictor is a machine learning application designed to forecast the likelihood of landslides based on various environmental factors. Use the sliders to input data and get predictions.")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = clean_data_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)

if __name__ == '__main__':
    main()
