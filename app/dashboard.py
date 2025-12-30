import streamlit as st
import requests
import pandas as pd
import os
import io
import sys

# HACK: Add project root to path so we can import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --------------------------------------------------------------------------------
# CONFIG & UTILS
# --------------------------------------------------------------------------------
st.set_page_config(page_title="CKD AI System", layout="wide", initial_sidebar_state="expanded")
API_URL = "http://127.0.0.1:8000"
DATA_DIR = "data/raw"

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["username"] == "admin" and st.session_state["password"] == "admin":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show inputs again
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct
        return True

# --------------------------------------------------------------------------------
# VIEWS
# --------------------------------------------------------------------------------

def doctor_view():
    st.title("👨‍⚕️ Doctor's Portal")
    st.markdown("### Patient Diagnostics & Risk Assessment")
    st.divider()

    # Sidebar for Input
    st.sidebar.header("Patient Vitals")
    
    # [NEW] File Uploader
    uploaded_report = st.sidebar.file_uploader("📂 Upload Patient Report (PDF)", type=["pdf"])
    
    pre_filled_data = {}
    if uploaded_report:
        from src.document_parser import DocumentParser
        parser = DocumentParser()
        with st.spinner("Parsing Report..."):
            extracted = parser.parse_pdf(uploaded_report.getvalue())
            if extracted:
                st.sidebar.success("✅ Data Extracted!")
                pre_filled_data = extracted
                
                # [NEW] Document Preview
                import base64
                st.markdown("### 📄 Document Preview")
                base64_pdf = base64.b64encode(uploaded_report.getvalue()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                st.info("👈 Compare extracted values in sidebar with this document.")
                
            else:
                st.sidebar.warning("Could not extract data. Please enter manually.")

    def user_input_features():
        data = {}
        
        # Helper to create an input field with an "Include" checkbox on the left
        def render_input(key, label, input_type, **kwargs):
            # Layout: [ Checkbox ] [ Input Field ]
            c1, c2 = st.sidebar.columns([0.15, 0.85])
            
            # 1. "Include" Checkbox (Left)
            # Default to Checked (Known) unless specifically set otherwise
            # [Refactor] Removed 'help' tooltip per request - cleaner UI
            is_included = c1.checkbox("", value=True, key=f"inc_{key}")
            
            # 2. Input Field (Right)
            # We get the default value from parsed report or passed default
            default_val = pre_filled_data.get(key, kwargs.get('value', kwargs.get('index', 0)))

            # If not included, disable the input
            kwargs['disabled'] = not is_included
            
            val = None
            
            # Logic to render specific types within col2
            with c2:
                if input_type == 'number':
                     # Ensure value is correct type
                    kwargs['value'] = default_val
                    val = st.number_input(label, **kwargs)
                elif input_type == 'slider':
                    kwargs['value'] = int(default_val)
                    val = st.slider(label, **kwargs)
                elif input_type == 'selectbox':
                    options = kwargs['options']
                    # Try to find default in options
                    idx = 0
                    val_str = str(default_val).lower()
                    
                    # Special handling for numeric selectboxes where default might be float/int
                    if isinstance(options[0], (int, float)):
                         if default_val in options:
                             idx = options.index(default_val)
                    else:
                        # String matching
                        for i, opt in enumerate(options):
                            if str(opt).lower() == val_str:
                                idx = i
                                break
                    
                    # Remove 'value' from kwargs as selectbox doesn't take it
                    if 'value' in kwargs:
                        del kwargs['value']

                    kwargs['index'] = idx
                    val = st.selectbox(label, **kwargs)
            
            # Return value only if included, else None
            return val if is_included else None
        
        # --- RENDER INPUTS ---
        
        # 1. Age
        data['age'] = render_input('age', "Age", 'number', min_value=1, max_value=120, value=60)
        
        # 2. Blood Pressure
        data['bp'] = render_input('bp', "Blood Pressure (bp)", 'slider', min_value=50, max_value=180, value=80)
        
        # 3. Specific Gravity
        data['sg'] = render_input('sg', "Specific Gravity (sg)", 'selectbox', options=[1.005, 1.010, 1.015, 1.020, 1.025], value=1.020)
        
        # 4. Albumin
        data['al'] = render_input('al', "Albumin (al)", 'selectbox', options=[0, 1, 2, 3, 4, 5], value=0)
        
        # 5. Sugar
        data['su'] = render_input('su', "Sugar (su)", 'selectbox', options=[0, 1, 2, 3, 4, 5], value=0)
        
        # 6. Red Blood Cells
        data['rbc'] = render_input('rbc', "Red Blood Cells (rbc)", 'selectbox', options=["normal", "abnormal"], value="normal")

        # 7. Pus Cell
        data['pc'] = render_input('pc', "Pus Cell (pc)", 'selectbox', options=["normal", "abnormal"], value="normal")
        
        # 8. Pus Cell Clumps
        data['pcc'] = render_input('pcc', "Pus Cell Clumps (pcc)", 'selectbox', options=["notpresent", "present"], value="notpresent")
        
        # 9. Bacteria
        data['ba'] = render_input('ba', "Bacteria (ba)", 'selectbox', options=["notpresent", "present"], value="notpresent")

        # 10. BGR
        data['bgr'] = render_input('bgr', "Blood Glucose Random (bgr)", 'number', min_value=50, max_value=500, value=120)

        # 11. Blood Urea
        data['bu'] = render_input('bu', "Blood Urea (bu)", 'number', min_value=10, max_value=300, value=40)

        # 12. Serum Creatinine
        data['sc'] = render_input('sc', "Serum Creatinine (sc)", 'number', min_value=0.4, max_value=15.0, value=1.2)

        # 13. Sodium
        data['sod'] = render_input('sod', "Sodium (sod)", 'number', min_value=100, max_value=160, value=135)

        # 14. Potassium
        data['pot'] = render_input('pot', "Potassium (pot)", 'number', min_value=2.0, max_value=7.0, value=4.0)

        # 15. Hemoglobin
        data['hemo'] = render_input('hemo', "Hemoglobin (hemo)", 'number', min_value=3.0, max_value=18.0, value=12.0)

        # 16. Packed Cell Volume
        data['pcv'] = render_input('pcv', "Packed Cell Volume (pcv)", 'number', min_value=15, max_value=60, value=40)

        # 17. White Blood Cell Count
        data['wc'] = render_input('wc', "White Blood Cell Count (wc)", 'number', min_value=2000, max_value=20000, value=8000)

        # 18. Red Blood Cell Count
        data['rc'] = render_input('rc', "Red Blood Cell Count (rc)", 'number', min_value=2.0, max_value=8.0, value=4.5)

        # 19. Hypertension
        data['htn'] = render_input('htn', "Hypertension (htn)", 'selectbox', options=["no", "yes"], value="no")

        # 20. Diabetes Mellitus
        data['dm'] = render_input('dm', "Diabetes Mellitus (dm)", 'selectbox', options=["no", "yes"], value="no")

        # 21. CAD
        data['cad'] = render_input('cad', "Coronary Artery Disease (cad)", 'selectbox', options=["no", "yes"], value="no")

        # 22. Appetite
        data['appet'] = render_input('appet', "Appetite (appet)", 'selectbox', options=["good", "poor"], value="good")

        # 23. Pedal Edema
        data['pe'] = render_input('pe', "Pedal Edema (pe)", 'selectbox', options=["no", "yes"], value="no")

        # 24. Anemia
        data['ane'] = render_input('ane', "Anemia (ane)", 'selectbox', options=["no", "yes"], value="no")

        return data

    input_data = user_input_features()
    
    # Confirmation View
    if uploaded_report:
        st.info("ℹ️ Values above were auto-filled from the uploaded report. Please review them in the sidebar before predicting.")

    col1, col2 = st.columns([1, 1])
    
    if st.button("Predict Risk", type="primary"):
        # 1. Prediction Request
        try:
            response = requests.post(f"{API_URL}/predict", json=input_data)
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to backend. Is the API server running?")
            response = None
            
        # 2. Process Response
        if response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    with col1:
                        st.subheader("Results")
                        risk_level = result["risk_level"]
                        color = "#ff4b4b" if risk_level == "High" else "#ffa421" if risk_level == "Medium" else "#21c354"
                        st.markdown(f"""
                            <div style="padding: 20px; border-radius: 10px; background-color: {color}20; border: 2px solid {color}; text-align: center;">
                                <h2 style="color: {color}; margin:0;">{risk_level} Risk</h2>
                                <p style="font-size: 1.2em; margin:0;">Probability: {result['probability']:.1%}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.subheader("🔍 Detailed AI Explanation")
                        if 'explanation' in result and result['explanation']:
                            # Convert to DataFrame for chart
                            exp_data = result['explanation']
                            # Sort by absolute importance
                            sorted_features = sorted(exp_data.items(), key=lambda x: abs(x[1]), reverse=True)
                            # Take top 10
                            top_features = dict(sorted_features[:10])
                            
                            st.write("Top factors driving this prediction:")
                            # Using a simple bar chart
                            st.bar_chart(top_features)
                        else:
                            st.info("No detailed explanation available.")
                    
                    # [NEW] Sensitivity Warnings
                    if "warnings" in result and result["warnings"]:
                        st.error("⚠️ CRITICAL MISSING DATA DETECTED")
                        for w in result["warnings"]:
                            st.write(w)
                        st.info("The AI simulated scenarios for these missing values and found they could change the diagnosis. It is recommended to test for these.")
                    else:
                        st.success("✅ Sensitivity Check Passed: Missing values did not affect the clinical outcome.")
                    
                    st.write("---")
                    st.write("Does this diagnosis align with your clinical assessment?")
                    c1, c2, c3 = st.columns([1,1,3])
                    if c1.button("✅ Yes"):
                        st.balloons()
                        st.success("Case logged as Correct Diagnosis.")
                    if c2.button("❌ No"):
                        # Send feedback
                        input_data['correct_diagnosis'] = 'notckd' if result['prediction'] == 'ckd' else 'ckd'
                        requests.post(f"{API_URL}/feedback", json=input_data)
                        st.warning("Feedback Sent. This case has been flagged for model retraining.")
                except Exception as e:
                    st.error(f"⚠️ An error occurred while rendering results: {e}")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")


def developer_view():
    st.title("💻 Developer Admin Portal")
    st.markdown("### Dataset Management & Model Retraining")
    
    if not check_password():
        st.stop()
        
    st.success("Authenticated as Admin")
    
    tab1, tab2 = st.tabs(["📂 Dataset Manager", "📊 Quality Audit"])
    
    with tab1:
        st.subheader("Add New Data")
        
        # Method 1: Upload
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"Preview ({len(df)} rows):")
            st.dataframe(df.head())
            if st.button("Save Uploaded Data"):
                path = os.path.join(DATA_DIR, uploaded_file.name)
                df.to_csv(path, index=False)
                st.success(f"Saved to {path}")
        
        st.write("---")
        
        # Method 2: URL
        st.subheader("Download from URL")
        url = st.text_input("Dataset URL (CSV)")
        if url and st.button("Download"):
            try:
                df = pd.read_csv(url)
                filename = url.split("/")[-1] or "downloaded_data.csv"
                path = os.path.join(DATA_DIR, filename)
                df.to_csv(path, index=False)
                st.success(f"Downloaded {len(df)} rows to {path}")
            except Exception as e:
                st.error(f"Failed to download: {e}")
                
        st.write("---")
        st.subheader("Existing Datasets")
        files = os.listdir(DATA_DIR)
        st.code("\n".join(files))

    with tab2:
        st.subheader("Data Quality Checker")
        selected_file = st.selectbox("Select File to Audit", files if 'files' in locals() else os.listdir(DATA_DIR))
        
        if selected_file:
            file_path = os.path.join(DATA_DIR, selected_file)
            if os.path.isfile(file_path): # Only process files
                df = pd.read_csv(file_path)
                
                # Check for duplicates
                dupes = df[df.duplicated()]
                num_dupes = len(dupes)
                
                # Check for nulls
                null_counts = df.isnull().sum().sum()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Rows", len(df))
                c2.metric("Duplicates", num_dupes, delta_color="inverse")
                c3.metric("Missing Values", null_counts, delta_color="inverse")
                
                if num_dupes > 0:
                    st.warning("⚠️ Duplicate Rows Found")
                    with st.expander("View Duplicates"):
                        st.dataframe(dupes)
                        
                if null_counts > 0:
                    st.warning("⚠️ Missing Values Detected")
                    st.bar_chart(df.isnull().sum())
                    
                st.write("### Data Preview")
                # Highlight corrupt/empty cells
                st.dataframe(df.style.highlight_null(color='red'))
            else:
                 st.info("Select a CSV file to audit.")

# --------------------------------------------------------------------------------
# MAIN ROUTER
# --------------------------------------------------------------------------------

def main():
    st.sidebar.title("Navigation")
    role = st.sidebar.radio("Select Role", ["Home", "Doctor", "Developer"])
    
    if role == "Home":
        st.header("Welcome to the CKD Prediction System")
        st.image("https://img.freepik.com/free-vector/medical-technology-concpet-with-monitor_23-2147677519.jpg", use_column_width=True) # Placeholder
        st.markdown("""
        **Select your portal from the sidebar:**
        - **Doctor**: Access the diagnostic tool.
        - **Developer**: Manage datasets and monitor system health.
        """)
        
    elif role == "Doctor":
        doctor_view()
        
    elif role == "Developer":
        developer_view()

if __name__ == "__main__":
    main()
