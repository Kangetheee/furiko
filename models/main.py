import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import pickle

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


# Main function
def main():
    data = clean_data()
    model, scaler = create_model(data)
    
    # Save the model and scaler using pickle
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()
