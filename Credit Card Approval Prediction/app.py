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
    # Collect data from the form
    gender = request.form.get('gender')
    owns_car = request.form.get('owns_car')
    number_of_children = int(request.form.get('number_of_children'))
    total_income = float(request.form.get('total_income'))
    income_type = request.form.get('income_type')
    education_type = request.form.get('education_type')
    family_status = request.form.get('family_status')
    housing_type = request.form.get('housing_type')
    day_of_birth = int(request.form.get('day_of_birth'))
    days_of_employment = int(request.form.get('days_of_employment'))
    occupation_type = request.form.get('occupation_type')
    family_member_number = int(request.form.get('family_member_number'))

    # Create a DataFrame from the form input
    input_data = pd.DataFrame({
        'CODE_GENDER': [gender],
        'FLAG_OWN_CAR': [owns_car],
        'CNT_CHILDREN': [number_of_children],
        'TOTAL_INCOME': [total_income],
        'NAME_INCOME_TYPE': [income_type],
        'NAME_EDUCATION_TYPE': [education_type],
        'NAME_FAMILY_STATUS': [family_status],
        'NAME_HOUSING_TYPE': [housing_type],
        'DAYS_BIRTH': [-day_of_birth],
        'DAYS_EMPLOYED': [-days_of_employment],
        'OCCUPATION_TYPE': [occupation_type],
        'CNT_FAM_MEMBERS': [family_member_number]
    })

    # Preprocessing: Apply binary encoding
    input_data = binary_encoder.transform(input_data)

    # Scale numeric columns
    input_data[['TOTAL_INCOME', 'DAYS_BIRTH', 'DAYS_EMPLOYED']] = scaler.transform(
        input_data[['TOTAL_INCOME', 'DAYS_BIRTH', 'DAYS_EMPLOYED']]
    )

    # Make prediction using the pre-trained model
    prediction = mlp_model.predict(input_data)

    # Return the prediction result as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
