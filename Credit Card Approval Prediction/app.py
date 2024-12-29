from flask import Flask, render_template, request, jsonify
from joblib import load
import pandas as pd
from sklearn.preprocessing import StandardScaler
from category_encoders import BinaryEncoder
from flask import Flask, render_template
app = Flask(__name__)

# Load the pre-trained model and preprocessing tools
mlp_model = load('mlp_model.joblib')  # Pre-trained MLP model
scaler = load('scaler.joblib')  # Pre-saved scaler
binary_encoder = load('binary_encoder.joblib')  # Pre-saved encoder

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data
        data = request.get_json()
        
        # Prepare the input for prediction
        input_data = pd.DataFrame([{
            "Gender": data["Radio"],
            "Owns Car": data["Radio1"],
            "Children": int(data["Number"]),
            "Income": float(data["Number2"]),
            "Work Status": data["Radio2"],
            "Education": data["Radio3"],
            "Marital Status": data["Radio4"],
            "Housing": data["Radio5"],
            "Birth Year": int(data["Date"]),
            "Application Year": int(data["Date1"]),
            "Occupation": data["Dropdown"],
            "Family Members": int(data["Number3"]),
        }])
        
        # Apply encoding and scaling
        input_encoded = binary_encoder.transform(input_data)
        input_scaled = scaler.transform(input_encoded)
        
        # Make prediction
        prediction = mlp_model.predict(input_scaled)[0]
        
        # Return the prediction
        return jsonify({"mlp_prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)