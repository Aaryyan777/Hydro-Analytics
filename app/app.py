
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

import os

app = Flask(__name__)

# Get the directory of the script
script_dir = os.path.dirname(__file__)

# Load the model and encoder
model = joblib.load(os.path.join(script_dir, 'water_quality_model_xgb.joblib'))
encoder = joblib.load(os.path.join(script_dir, 'encoder.joblib'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Create a dataframe from the input
    input_data = {
        'Sample class': [data['sample_class']],
        'Residual Free Chlorine (mg/L)': [float(data['chlorine'])],
        'Turbidity (NTU)': [float(data['turbidity'])],
        'Sample_Month': [int(data['month'])],
        'Sample_DayOfWeek': [int(data['dayofweek'])],
        'Sample_Hour': [int(data['hour'])]
    }
    df = pd.DataFrame(input_data)

    # One-hot encode the 'Sample class' column
    encoded_data = encoder.transform(df[['Sample class']])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Sample class']))
    df = pd.concat([df.drop(columns=['Sample class']), encoded_df], axis=1)

    # Make prediction
    prediction = model.predict(df)[0]

    # Return the prediction
    result = {'prediction': 'Deterioration' if prediction == 1 else 'Normal'}
    return jsonify(result)

# Load the Isolation Forest model
iso_forest_model = joblib.load('isolation_forest_model.joblib')

# Load the preprocessed Brisbane data to get the correct column order and for mean imputation
brisbane_df = pd.read_csv(r"C:\Users\DELL\Desktop\predict\brisbane-water-quality-preprocessed.csv")
brisbane_cols = brisbane_df.drop(columns=['Deterioration']).columns

@app.route('/predict_anomaly_brisbane', methods=['POST'])
def predict_anomaly_brisbane():
    data = request.get_json()

    # Create a dataframe from the input, using means for missing values
    # This is a simplified approach for the web app. In a real-world scenario,
    # all features would likely be available.
    input_data = {}
    for col in brisbane_cols:
        # For this example, we'll assume the keys in `data` match the column names
        # for the fields we are getting from the form.
        if col in data:
            input_data[col] = [float(data[col])]
        else:
            input_data[col] = [brisbane_df[col].mean()]
    
    df = pd.DataFrame(input_data)
    df = df[brisbane_cols] # Ensure correct column order

    # Make prediction
    prediction = iso_forest_model.predict(df)[0]

    # Return the prediction
    result = {'prediction': 'Anomaly' if prediction == -1 else 'Normal'}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
