import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Configure Streamlit page
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model_data = load_model()
model = model_data['model']
features = model_data['features']
edu_map = model_data['edu_map']
emp_map = model_data['emp_map']

# Custom CSS for modern UI
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 1.1rem;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(52, 152, 219, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 6px 8px rgba(52, 152, 219, 0.4);
        transform: translateY(-2px);
    }
    .result-text-approved {
        color: #27ae60;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin-top: 1rem;
    }
    .result-text-rejected {
        color: #e74c3c;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin-top: 1rem;
    }
    div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label {
        color: #34495e;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏦 Modern Loan Approval System</div>', unsafe_allow_html=True)

# Image Paths
APPROVED_IMG_PATH = "assets/approved.jpg"
REJECTED_IMG_PATH = "assets/rejected.jpg"

# Form container
with st.container():
    st.markdown("### Applicant Information")
    
    col1, col2 = st.columns(2)
    with col1:
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)
        education = st.selectbox("Education Level", options=["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed?", options=["No", "Yes"])
        cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
        loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=10)

    with col2:
        income_annum = st.number_input("Annual Income (₹)", min_value=0, value=500000, step=50000)
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=1000000, step=100000)
        residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=2000000, step=100000)
        commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, value=0, step=100000)
        luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, value=0, step=100000)
        bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, value=500000, step=50000)

    st.markdown("---")

    if st.button("Predict Loan Status"):
        # Map categorical variables
        edu_val = edu_map[education]
        emp_val = emp_map[self_employed]
        
        # Prepare input array matching 'features'
        input_data = {
            'no_of_dependents': no_of_dependents,
            'education': edu_val,
            'self_employed': emp_val,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets_value,
            'commercial_assets_value': commercial_assets_value,
            'luxury_assets_value': luxury_assets_value,
            'bank_asset_value': bank_asset_value
        }
        
        # Ensure order matches
        input_df = pd.DataFrame([input_data])[features]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display Result
        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns([1,2,1])
        with res_col2:
            if prediction == 1:
                st.markdown('<div class="result-text-approved">🎉 Congratulations! Loan Approved!</div>', unsafe_allow_html=True)
                try:
                    img = Image.open(APPROVED_IMG_PATH)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.success("Your loan application has been APPROVED.")
            else:
                st.markdown('<div class="result-text-rejected">❌ We are sorry. Loan Rejected.</div>', unsafe_allow_html=True)
                try:
                    img = Image.open(REJECTED_IMG_PATH)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error("Your loan application has been REJECTED.")
