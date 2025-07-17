from flask import Flask, request, render_template
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load('Linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    features = [
        float(request.form['MedInc']),
        float(request.form['HouseAge']),
        float(request.form['AveRooms']),
        float(request.form['AveBedrms']),
        float(request.form['Population']),
        float(request.form['AveOccup']),
        float(request.form['Latitude']),
        float(request.form['Longitude'])
    ]
    
    # Scale features
    scaled_features = scaler.transform([features])
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    
    return render_template('index.html', prediction_text=f"Predicted House Value: {prediction:.2f} (in $100k)")

if __name__ == '__main__':
    app.run(debug=True)
