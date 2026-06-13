import os
import io
import json
import joblib
import datetime
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Ensure models are loaded at startup
MODELS_DIR = "models"
MODELS_LOADED = False

svm_model = None
svm_scaler = None
knn_model = None
knn_scaler = None
gb_model = None
gb_scaler = None

jfo_model = None
jfo_scaler = None
jfo_features = None

metrics_data = None

def load_models():
    global svm_model, svm_scaler, knn_model, knn_scaler, gb_model, gb_scaler
    global jfo_model, jfo_scaler, jfo_features, metrics_data, MODELS_LOADED
    
    try:
        svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.joblib"))
        svm_scaler = joblib.load(os.path.join(MODELS_DIR, "svm_scaler.joblib"))
        
        knn_model = joblib.load(os.path.join(MODELS_DIR, "knn_model.joblib"))
        knn_scaler = joblib.load(os.path.join(MODELS_DIR, "knn_scaler.joblib"))
        
        gb_model = joblib.load(os.path.join(MODELS_DIR, "gb_model.joblib"))
        gb_scaler = joblib.load(os.path.join(MODELS_DIR, "gb_scaler.joblib"))
        
        jfo_model = joblib.load(os.path.join(MODELS_DIR, "jfo_model.joblib"))
        jfo_scaler = joblib.load(os.path.join(MODELS_DIR, "jfo_scaler.joblib"))
        jfo_features = joblib.load(os.path.join(MODELS_DIR, "jfo_features.joblib"))
        
        with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
            metrics_data = json.load(f)
            
        MODELS_LOADED = True
        print("All machine learning models and scalers loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        MODELS_LOADED = False

# Try to load models on startup (if already trained)
if os.path.exists(os.path.join(MODELS_DIR, "metrics.json")):
    load_models()

@app.route("/")
def index():
    # Attempt to reload if not loaded
    if not MODELS_LOADED:
        load_models()
    return render_template("index.html")

@app.route("/metrics", methods=["GET"])
def get_metrics():
    if not MODELS_LOADED:
        load_models()
    if metrics_data:
        return jsonify(metrics_data)
    else:
        return jsonify({"error": "Metrics not available. Please run train_models.py first."}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if not MODELS_LOADED:
        load_models()
        if not MODELS_LOADED:
            return jsonify({"error": "Models not loaded. Train models first."}), 500
            
    try:
        data = request.json
        age = int(data.get("age", 45))
        sex = int(data.get("sex", 1))
        cp = int(data.get("cp", 2))
        trestbps = int(data.get("trestbps", 130))
        chol = int(data.get("chol", 240))
        model_name = data.get("model", "svm").lower()
        
        # Validations
        if age < 1 or age > 120:
            return jsonify({"error": "Invalid age range"}), 400
        if trestbps < 80 or trestbps > 220:
            return jsonify({"error": "Invalid blood pressure range"}), 400
        if chol < 100 or chol > 600:
            return jsonify({"error": "Invalid cholesterol range"}), 400

        # Construct features input vector
        # Standard models: age, sex, cp, trestbps, chol
        base_features_vector = [age, sex, cp, trestbps, chol]
        
        if model_name == "svm":
            scaled_features = svm_scaler.transform([base_features_vector])
            prediction_class = svm_model.predict(scaled_features)[0]
            probabilities = svm_model.predict_proba(scaled_features)[0]
        elif model_name == "knn":
            scaled_features = knn_scaler.transform([base_features_vector])
            prediction_class = knn_model.predict(scaled_features)[0]
            probabilities = knn_model.predict_proba(scaled_features)[0]
        elif model_name == "gb":
            scaled_features = gb_scaler.transform([base_features_vector])
            prediction_class = gb_model.predict(scaled_features)[0]
            probabilities = gb_model.predict_proba(scaled_features)[0]
        elif model_name == "jfo":
            # SVM + JFO was trained on top 4 features dynamically selected by Chi-Square.
            # Map input parameters to selected features
            features_dict = {"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol}
            selected_features_vector = [features_dict[feat] for feat in jfo_features]
            
            scaled_features = jfo_scaler.transform([selected_features_vector])
            prediction_class = jfo_model.predict(scaled_features)[0]
            probabilities = jfo_model.predict_proba(scaled_features)[0]
        else:
            return jsonify({"error": "Invalid model selection"}), 400
            
        # Class 1 = Disease Detected, Class 0 = Low Risk
        if prediction_class == 1:
            prediction_str = "Heart Disease Detected"
            confidence = float(probabilities[1] * 100)
        else:
            prediction_str = "Low Risk"
            confidence = float(probabilities[0] * 100)
            
        return jsonify({
            "prediction": prediction_str,
            "confidence": confidence
        })
        
    except Exception as e:
        print(f"Error during prediction API: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/export", methods=["GET"])
def export_pdf():
    try:
        # Import reportlab libraries inside route to avoid hard dependencies if not installed yet
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    except ImportError:
        return jsonify({"error": "ReportLab library not installed on server. Cannot export PDF."}), 500
        
    try:
        # Retrieve query parameters
        age = request.args.get("age", "45")
        sex_code = request.args.get("sex", "1")
        cp_code = request.args.get("cp", "2")
        trestbps = request.args.get("trestbps", "130")
        chol = request.args.get("chol", "240")
        model_code = request.args.get("model", "svm")
        prediction = request.args.get("pred", "Low Risk")
        confidence = request.args.get("conf", "95.0")
        
        # Human-readable mappings
        sex_str = "Male" if sex_code == "1" else "Female"
        
        cp_map = {
            "0": "Typical Angina (value 0)",
            "1": "Atypical Angina (value 1)",
            "2": "Non-Anginal Pain (value 2)",
            "3": "Asymptomatic (value 3)"
        }
        cp_str = cp_map.get(cp_code, "Unknown")
        
        model_map = {
            "svm": "Support Vector Machine (SVM)",
            "knn": "K-Nearest Neighbors (KNN)",
            "gb": "Gradient Boosting Classifier",
            "jfo": "SVM + Jellyfish Optimization (JFO)"
        }
        model_str = model_map.get(model_code, model_code.upper())
        
        # Setup PDF document buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=letter,
            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=15,
            alignment=1  # Centered
        )
        
        section_heading = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=colors.HexColor('#0d9488'),
            spaceBefore=12,
            spaceAfter=8
        )
        
        normal_style = styles['Normal']
        normal_bold = ParagraphStyle(
            'NormalBold', parent=normal_style, fontName='Helvetica-Bold'
        )
        
        result_style = ParagraphStyle(
            'ResultStyle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=colors.HexColor('#ef4444') if "Detected" in prediction else colors.HexColor('#10b981')
        )
        
        story = []
        
        # Title Header
        story.append(Paragraph("HeartAI Diagnostic Center", title_style))
        story.append(Paragraph("AI-Powered Clinical Screening Report", ParagraphStyle('Sub', parent=normal_style, alignment=1, fontSize=10, textColor=colors.HexColor('#64748b'))))
        story.append(Spacer(1, 15))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceBefore=1, spaceAfter=15))
        
        # Report Metadata
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_data = [
            [Paragraph("<b>Report Date:</b>", normal_style), Paragraph(current_time, normal_style),
             Paragraph("<b>Diagnostic ID:</b>", normal_style), Paragraph(f"HA-{hash(current_time) % 1000000:06d}", normal_style)]
        ]
        t_meta = Table(metadata_data, colWidths=[90, 180, 90, 180])
        t_meta.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(t_meta)
        story.append(Spacer(1, 15))
        
        # 1. Patient Demographics Section
        story.append(Paragraph("Patient Clinical Records", section_heading))
        patient_data = [
            [Paragraph("Clinical Attribute", normal_bold), Paragraph("Observed Value", normal_bold), Paragraph("Reference Standard", normal_bold)],
            [Paragraph("Age", normal_style), Paragraph(f"{age} years", normal_style), Paragraph("-", normal_style)],
            [Paragraph("Sex", normal_style), Paragraph(sex_str, normal_style), Paragraph("-", normal_style)],
            [Paragraph("Chest Pain Type (cp)", normal_style), Paragraph(cp_str, normal_style), Paragraph("Asymptomatic / Typical / Atypical", normal_style)],
            [Paragraph("Resting Blood Pressure", normal_style), Paragraph(f"{trestbps} mm Hg", normal_style), Paragraph("Normal: < 120 mm Hg", normal_style)],
            [Paragraph("Serum Cholesterol (chol)", normal_style), Paragraph(f"{chol} mg/dl", normal_style), Paragraph("Desirable: < 200 mg/dl", normal_style)],
        ]
        t_patient = Table(patient_data, colWidths=[180, 180, 180])
        t_patient.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eff6ff')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t_patient)
        story.append(Spacer(1, 15))
        
        # 2. Predictive Assessment Section
        story.append(Paragraph("Artificial Intelligence Diagnostic Assessment", section_heading))
        result_data = [
            [Paragraph("Evaluation Metric", normal_bold), Paragraph("Prediction Value", normal_bold)],
            [Paragraph("Selected Classifier", normal_style), Paragraph(model_str, normal_style)],
            [Paragraph("Diagnostic Status", normal_style), Paragraph(prediction, result_style)],
            [Paragraph("Risk Score / Confidence", normal_style), Paragraph(f"{confidence}%", normal_bold)],
        ]
        t_result = Table(result_data, colWidths=[200, 340])
        t_result.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0fdfa')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(t_result)
        story.append(Spacer(1, 20))
        
        # 3. Clinical Disclaimer & Notes
        story.append(Paragraph("Clinical Summary & Disclaimer", section_heading))
        disclaimer_text = (
            "<b>Important Clinical Notice:</b> This report is generated by an automated Artificial Intelligence diagnostic assistant trained "
            "on clinical cardiovascular datasets. It uses advanced classification algorithms (including support vector machine learning "
            "parameterized with Jellyfish Optimization heuristics). While these models achieve high training accuracy on reference sets, "
            "this screening analysis does NOT constitute definitive medical advice or diagnosis. It is intended strictly as a decision-support "
            "tool to assist clinicians. Results should be interpreted by a board-certified cardiologist in conjunction with further diagnostic tests "
            "(such as ECGs, angiograms, and echocardiograms)."
        )
        story.append(Paragraph(disclaimer_text, ParagraphStyle('DisclaimerStyle', parent=normal_style, fontSize=9, textColor=colors.HexColor('#475569'), leading=12)))
        story.append(Spacer(1, 25))
        
        # 4. Signature Block & Attribution
        sig_data = [
            [Paragraph('<b>Physician Signature:</b>', normal_style), Paragraph('<b>Developed by:</b>', normal_style)],
            [Paragraph('_____________________________<br/><br/><small class="text-muted">MD Cardiology / Reviewer</small>', normal_style),
             Paragraph('<b>Thirunaavukkarasu S</b><br/>B.Tech Information Technology<br/>Dr. NGP Institute of Technology', normal_style)]
        ]
        t_sig = Table(sig_data, colWidths=[270, 270])
        t_sig.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(t_sig)
        
        # 5. Footer Line
        story.append(Spacer(1, 30))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceBefore=1, spaceAfter=10))
        story.append(Paragraph("This project was independently designed, developed, trained, tested, and implemented by Thirunaavukkarasu S.", 
                               ParagraphStyle('Foot', parent=normal_style, alignment=1, fontSize=8, textColor=colors.HexColor('#94a3b8'))))
        
        doc.build(story)
        buffer.seek(0)
        
        # Clean filename
        safe_filename = f"Heart_Disease_Report_{age}_{sex_str}.pdf"
        return send_file(
            buffer,
            as_attachment=True,
            download_name=safe_filename,
            mimetype="application/pdf"
        )
        
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        return jsonify({"error": f"Failed to generate PDF report: {str(e)}"}), 500

if __name__ == "__main__":
    # Load models on start
    load_models()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
