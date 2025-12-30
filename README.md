# Chronic Kidney Disease (CKD) Prediction System 🏥

An advanced AI-powered system designed to assist medical professionals in the early detection and risk assessment of Chronic Kidney Disease. This project leverages Machine Learning (Random Forest) for high-accuracy predictions and SHAP (SHapley Additive exPlanations) for transparent, "white-box" explainability.

## 🌟 Key Features

### 👨‍⚕️ Doctor's Portal

- **Real-time Risk Prediction**: Instant probability assessment (High/Medium/Low Risk).
- **📄 PDF Report Parser**: Auto-fill patient details by uploading a standard medical report (PDF).
- **👁️ Document Preview**: Compare extracted data side-by-side with the original report.
- **✅ Intelligent Form**: "Unknown" checkboxes allow for sensitivity analysis on missing data (disabled inputs are simulated by the AI).
- **🔍 Detailed AI Explanations**: Visual bar charts showing _exactly_ why a prediction was made (e.g., "High Blood Pressure contributed +15% to risk").

### 💻 Developer/Admin Portal

- **Dataset Management**: Upload new CSV datasets or download from URLs.
- **Data Quality Audit**: Automatically detect duplicates and missing values.
- **Feedback Loop**: Retrain models based on doctor feedback.

### ⚙️ Backend & Architecture

- **FastAPI**: High-performance API handling predictions and SHAP calculations.
- **Streamlit**: Interactive, user-friendly dashboard for doctors and admins.
- **Robust Pipeline**: Automated module for data cleaning, preprocessing, and model serialization.

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.10+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/mhd-hashir/Chronic_kidney_disease_predictor.git
cd Chronic_kidney_disease_predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

We have provided a convenient launcher script for Windows.
Simply double-click **`execute.bat`** or run:

```bash
./execute.bat
```

This will automatically:

1.  Start the FastAPI Backend (Port 8000)
2.  Launch the Streamlit Dashboard (Port 8501)

---

## 📂 Project Structure

```
├── app/
│   ├── api.py               # FastAPI Backend & Endpoints
│   └── dashboard.py         # Streamlit Frontend (Doctor & Admin Views)
├── data/                    # Raw and Processed Datasets
├── models/                  # Trained Models (.pkl) and Data Stats
├── src/
│   ├── document_parser.py   # PDF Extraction Logic
│   ├── explainability.py    # SHAP / LIME Explainability Module
│   └── preprocessing.py     # Data Cleaning & Transformation Pipelines
├── execute.bat              # One-click Launcher Script
└── requirements.txt         # Project Dependencies
```

## 🤖 Tech Stack

- **ML Engine**: Scikit-Learn (Random Forest Classifier)
- **Explainability**: SHAP (Shapley Additive explanations)
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy, PyPDF

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
