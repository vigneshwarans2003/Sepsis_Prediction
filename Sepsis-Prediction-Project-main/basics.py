from flask import Flask, request, redirect, url_for, render_template, jsonify, send_file
import pickle
import numpy as np
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph,Spacer
from reportlab.lib.units import inch  
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import base64
from io import BytesIO



app = Flask(__name__)



# Load models from pickle file
model_file_path = 'blending_classifier_models.pkl'
with open(model_file_path, 'rb') as f:
    loaded_models = pickle.load(f)

catboost = loaded_models['catboost']
xgb_model = loaded_models['xgb']
rf = loaded_models['rf']
meta_learner_blend = loaded_models['meta_learner_blend']

# Feature names corresponding to the model input
feature_names = [
    'HospAdmTime', 'Age', 'ICULOS', 'Hour', 'HR', 'PatientID', 'DBP', 'SBP',
    'Resp', 'Temp', 'MAP', 'Unit2', 'O2Sat', 'Gender',
    'Unit1', 'FiO2', 'EtCO2'
]

def predict_sepsis(features):
    # Convert input features to DataFrame for model prediction
    input_data = pd.DataFrame([features], columns=feature_names)
    # Predict probabilities with each model
    preds_catboost = catboost.predict_proba(input_data)[:, 1]
    preds_xgb = xgb_model.predict_proba(input_data)[:, 1]
    preds_rf = rf.predict_proba(input_data)[:, 1]
    # Stack predictions and use the meta-learner to make final prediction
    preds_blend = np.column_stack((preds_catboost, preds_xgb, preds_rf))
    final_prediction = meta_learner_blend.predict(preds_blend)[0]
    final_probability = meta_learner_blend.predict_proba(preds_blend)[0, 1]
    return final_prediction, final_probability

@app.route('/')
def index():
    return render_template('Login.html')

@app.route('/home')
def home():
    return render_template('Login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    if username == 'Karthi' and password == 'KK':
        return redirect(url_for('new'))
    else:
        return "Invalid username or password. Please try again."

@app.route('/new')
def new():
    return render_template('new.html')

@app.route('/web1')
def web1():
    return render_template('web1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data from the HTML page
        features = [
            float(request.form.get('a', 0)), float(request.form.get('b', 0)), float(request.form.get('c', 0)), float(request.form.get('d', 0)),
            float(request.form.get('e', 0)), float(request.form.get('f', 0)), float(request.form.get('g', 0)), float(request.form.get('h', 0)),
            float(request.form.get('i', 0)), float(request.form.get('j', 0)), float(request.form.get('k', 0)), float(request.form.get('l', 0)),
            float(request.form.get('m', 0)), float(request.form.get('o', 0)), float(request.form.get('p', 0)), float(request.form.get('q', 0)),
            float(request.form.get('r', 0))
        ]

        # Get the prediction and probability
        prediction, probability = predict_sepsis(features)
        prediction_label = 'Yes' if prediction == 1 else 'No'

        # Render the result in the template
        return render_template('after.html', prediction=prediction_label, probability=probability)

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    try:
        # Collect form data from the HTML page
        features = {
            'HospAdmTime': float(request.form.get('a', 0)),
            'Age': float(request.form.get('b', 0)),
            'ICULOS': float(request.form.get('c', 0)),
            'Hour': float(request.form.get('d', 0)),
            'HR': float(request.form.get('e', 0)),
            'PatientID': float(request.form.get('f', 0)),
            'DBP': float(request.form.get('g', 0)),
            'SBP': float(request.form.get('h', 0)),
            'Resp': float(request.form.get('i', 0)),
            'Temp': float(request.form.get('j', 0)),
            'MAP': float(request.form.get('k', 0)),
            'Unit2': float(request.form.get('l', 0)),
            'O2Sat': float(request.form.get('m', 0)),
            'Gender': float(request.form.get('o', 0)),
            'Unit 1': float(request.form.get('p', 0)),
            'FiO2': float(request.form.get('q', 0)),
            'EtCO2': float(request.form.get('r', 0)),
        }

        # Normal ranges for the parameters
        normal_ranges = {
            'HR': (60, 100),
            'DBP': (60, 80),
            'SBP': (90, 120),
            'Resp': (12, 16),
            'Temp': (36.5, 37.5),
            'O2Sat': (95, 100),
            'FiO2': (0.21, 0.5),
            'EtCO2': (35, 45),
            'MAP': (70, 100)
        }

        # Assuming predict_sepsis returns a tuple of (prediction, probability)
        prediction, probability = predict_sepsis(list(features.values()))
        prediction_label = 'Yes' if prediction == 1 else 'No'
        
        # Determine sepsis severity
        if probability >= 0.7:
            condition_severity = 'Serious'
        elif 0.4 <= probability < 0.7:
            condition_severity = 'Mild'
        else:
            condition_severity = 'No Sepsis'

        # Generate the PDF
        buffer = io.BytesIO()
        pdf = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []

        # Title without any free space above
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        title_style.spaceBefore = 0
        title = Paragraph("MEDICAL REPORT", title_style)
        elements.append(title)

        # Centered Sepsis Diagnosis information
        sepsis_info = Paragraph(f"<b>Sepsis Diagnosis:</b> {prediction_label} ({condition_severity} Condition)", styles['Title'])
        elements.append(sepsis_info)
        elements.append(Spacer(1, 12))

        # Table data
        table_data = [['Parameter', 'Value', 'Normal Range', 'Condition']]
        high_values = []
        low_values = []

        for key, value in features.items():
            if key in normal_ranges:
                low, high = normal_ranges[key]
                condition = 'Normal'
                if float(value) < low:
                    condition = 'Low'
                    low_values.append(key)
                elif float(value) > high:
                    condition = 'High'
                    high_values.append(key)
                table_data.append([key, value, f"{low}-{high}", condition])
            else:
                table_data.append([key, value, 'N/A', 'N/A'])

        # Add prediction and probability to the table
        table_data.append(['Sepsis Prediction', prediction_label, 'N/A', 'N/A'])
        table_data.append(['Sepsis Probability', f"{probability:.2f}", 'N/A', 'N/A'])

        # Create the table
        table = Table(table_data, colWidths=[2 * inch] * 4)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.gray),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        # Feedback section
        feedback = []

        # Heart-related feedback
        if any(param in high_values + low_values for param in ['HR', 'SBP', 'DBP', 'MAP', 'Temp']):
            reason = ', '.join([f"high {param}" if param in high_values else f"low {param}" for param in ['HR', 'SBP', 'DBP', 'MAP', 'Temp'] if param in high_values + low_values])
            feedback.append(f"1. You might get heart diseases, go for a heart checkup.")
            feedback.append(f"   Reason: Due to {reason}.")
            feedback.append(f"   Remedy: Do exercise. Take healthy foods, reduce cholesterol and fatty foods.")

        # Lung-related feedback
        if any(param in high_values + low_values for param in ['Resp', 'O2Sat', 'FiO2', 'EtCO2']):
            reason = ', '.join([f"high {param}" if param in high_values else f"low {param}" for param in ['Resp', 'O2Sat', 'FiO2', 'EtCO2'] if param in high_values + low_values])
            feedback.append(f"2. You might get lung diseases like pneumonia.")
            feedback.append(f"   Reason: Due to {reason}.")
            feedback.append(f"   Remedy: Do Yoga, Go for a walk, avoid pollution.")

        # Kidney-related feedback
        if any(param in high_values + low_values for param in ['MAP', 'SBP', 'DBP']):
            reason = ', '.join([f"high {param}" if param in high_values else f"low {param}" for param in ['MAP', 'SBP', 'DBP'] if param in high_values + low_values])
            feedback.append(f"3. You might get kidney dysfunction.")
            feedback.append(f"   Reason: Due to {reason}.")
            feedback.append(f"   Remedy: Drink more water, try avoiding direct sun rays, avoid junk foods.")

        if feedback:
            feedback_title = Paragraph("<b>FEEDBACKS:</b>", styles['Heading2'])
            elements.append(feedback_title)
            elements.append(Spacer(1, 12))
            for item in feedback:
                elements.append(Paragraph(item, styles['Normal']))
                elements.append(Spacer(1, 6))

        # Build the PDF
        pdf.build(elements)

        # Move to the beginning of the buffer
        buffer.seek(0)

        return send_file(buffer, as_attachment=True, download_name='medical_report.pdf', mimetype='application/pdf')

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        # Collect form data from the request
        features = {
            'HR': float(request.form.get('e', 0)),
            'DBP': float(request.form.get('g', 0)),
            'SBP': float(request.form.get('h', 0)),
            'Resp': float(request.form.get('i', 0)),
            'Temp': float(request.form.get('j', 0)),
            'O2Sat': float(request.form.get('m', 0)),
            'FiO2': float(request.form.get('q', 0)),
            'EtCO2': float(request.form.get('r', 0)),
            'MAP': float(request.form.get('k', 0))
        }

        # Normal ranges for the parameters
        normal_ranges = {
            'HR': (60, 100),
            'DBP': (60, 80),
            'SBP': (90, 120),
            'Resp': (12, 16),
            'Temp': (36.5, 37.5),
            'O2Sat': (95, 100),
            'FiO2': (0.21, 0.5),
            'EtCO2': (35, 45),
            'MAP': (70, 100)
        }

        # Define colors for the bars based on the range
        colors = []
        for key, value in features.items():
            if key in normal_ranges:
                low, high = normal_ranges[key]
                if value < low:
                    colors.append('yellow')
                elif value > high:
                    colors.append('red')
                else:
                    colors.append('green')

        # Create the bar chart
        plt.figure(figsize=(8, 4))
        plt.bar(features.keys(), features.values(), color=colors)
        plt.xlabel('Parameters')
        plt.ylabel('Values')
        plt.title('Patient Input Values')

        # Save the plot to a BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        # Encode the image to base64
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        
        return jsonify({'chart': f"data:image/png;base64,{img_base64}"})

    except Exception as e:
        return jsonify({'error': str(e)})
    

if __name__ == '__main__':
    app.run(debug=True)
