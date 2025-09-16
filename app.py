import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and scaler
model = load_model('stress_level_model.h5')  # Ensure this is the model you trained
scaler = StandardScaler()

# Load dataset to get scaling values (for standardization)
data = pd.read_csv("Stress-Lysis.csv")  # Updated dataset name
X = data.drop(columns=['Stress Level'])
scaler.fit(X)

# Load the label encoder for stress levels
y = data['Stress Level']
label_encoder = LabelEncoder()
label_encoder.fit(y)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        humidity = float(request.form['humidity'])
        temperature = float(request.form['temperature'])
        step_count = float(request.form['step_count'])
        
        # Prepare input data
        input_data = np.array([[humidity, temperature, step_count]])
        input_scaled = scaler.transform(input_data)
        
        # Get prediction from the model
        prediction = model.predict(input_scaled)
        stress_level = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        
        # Define description for each stress level
        descriptions = {
            0: "Low stress level indicates that the individual is calm and relaxed. Their physiological parameters such as humidity, temperature, and step count show normal patterns of behavior.",
            1: "Normal stress level means the individual is in a balanced state, neither overly stressed nor under active. The physiological parameters are within standard ranges.",
            2: "High stress level indicates that the individual is experiencing significant stress. This is reflected by changes in physiological parameters such as an increase in temperature and humidity, and a decrease in step count."
        }

        # Get the description based on the predicted stress level
        stress_description = descriptions[int(stress_level)]
        
        return render_template('index.html', prediction_text=f"Predicted Stress Level: {stress_level}", stress_description=stress_description)

if __name__ == '__main__':
    app.run(debug=True)
