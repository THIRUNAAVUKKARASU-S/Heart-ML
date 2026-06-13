# Heart Disease Prediction using Machine Learning

An AI-powered clinical diagnostics web application designed to predict the likelihood of heart disease in real time. The system features a responsive medical dashboard (Light/Dark mode) that accepts patient records and evaluates them against multiple trained machine learning models, displaying immediate confidence metrics, a risk progress meter, and supporting dynamic PDF clinical report downloads.

---

## 🌟 Key Features

*   **Real-Time Predictions:** Updates risk diagnostics dynamically as patient records are entered, using debounced asynchronous requests (no page refreshes).
*   **Multiple ML Classifiers:** Compares outcomes across four models:
    *   Support Vector Machine (Linear SVM)
    *   K-Nearest Neighbors (KNN)
    *   Gradient Boosting Classifier
    *   SVM with **Jellyfish Optimization (JFO)** and Chi-Square feature selection.
*   **Class Balancing:** Applied **SMOTE** oversampling on scaled clinical datasets to prevent minority group predictive bias.
*   **Medical UI Design:** Sleek modern interface built on Bootstrap 5 and custom glassmorphism styles with a system-wide Light and Dark mode toggle.
*   **Performance Charts:** Renders dynamic comparative metrics (Bar & Radar charts of Accuracy, Precision, Recall, and F1-score) and confidence distributions via Chart.js.
*   **Export Clinical PDF Reports:** Compiles formatted screening PDF reports (using ReportLab) complete with patient demographics, classifier details, cardiologist signature lines, and developer attribution.

---

## 📊 Model Performance Summary

The classifiers evaluated on the balanced test partition achieved the following metrics:

| Model Classifier | Accuracy | Precision | Recall | F1-Score | Features Used |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SVM (Linear)** | 79.6% | 86.3% | 75.2% | 80.4% | `['age', 'sex', 'cp', 'trestbps', 'chol']` |
| **K-Nearest Neighbors** | 83.4% | 87.3% | 82.1% | 84.6% | `['age', 'sex', 'cp', 'trestbps', 'chol']` |
| **Gradient Boosting** | 88.6% | 90.4% | 88.9% | 89.7% | `['age', 'sex', 'cp', 'trestbps', 'chol']` |
| **SVM + JFO** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | `['age', 'cp', 'trestbps', 'chol']` (Chi-Square top 4) |

*Note: The Jellyfish Optimization (JFO) algorithm successfully optimized the hyperparameters (C, gamma) of an RBF kernel SVM to achieve optimal convergence and a perfect test fit.*

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git installed on your system

### 1. Clone the Repository
```bash
git clone https://github.com/THIRUNAAVUKKARASU-S/Heart-ML.git
cd Heart-ML
```

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn imbalanced-learn joblib flask flask-cors reportlab
```

### 3. Train the Machine Learning Models
Execute the training script to preprocess the dataset, balance the target categories, run the JFO optimization, and serialize the trained models and scalers:
```bash
python train_models.py
```
*This will create a `models/` directory containing the saved `.joblib` objects and `metrics.json` evaluation file.*

### 4. Launch the Web Server
Start the Flask application:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to interact with the screening system.

---

## 🏥 Clinical Disclaimer
This application is designed strictly as a clinical decision-support screening tool. Machine learning predictions are trained on historical reference datasets and do not constitute formal medical diagnoses or advice. All predictive outputs must be interpreted by a board-certified cardiologist in conjunction with full diagnostics (ECGs, angiograms, echocardiograms).

---

## 👨‍💻 Developer Information

*   **Name:** Thirunaavukkarasu S
*   **Degree:** B.Tech Information Technology
*   **Institution:** Dr. NGP Institute of Technology
*   **Skills:** Machine Learning, Artificial Intelligence, Python, MERN Stack, Data Science
*   **Project Title:** Heart Disease Prediction using Machine Learning

*"This project was independently designed, developed, trained, tested, and implemented by Thirunaavukkarasu S."*
