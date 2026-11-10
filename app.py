import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import datetime

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Detection & Diagnostic System",
    page_icon="🩺",
    layout="centered"
)

# Custom Clinical CSS Styling
st.markdown("""
<style>
    /* Global styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 800px;
    }
    
    /* Centered Header */
    .main-header {
        text-align: center;
        padding-bottom: 12px;
        margin-bottom: 24px;
        border-bottom: 2px solid #E2E8F0;
    }
    .main-header h1 {
        color: #1E3A8A;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #64748B;
        font-size: 14px;
        margin: 0;
    }

    /* Section Headers */
    .section-title {
        font-size: 15px;
        font-weight: 600;
        text-transform: uppercase;
        color: #1E3A8A;
        letter-spacing: 0.5px;
        margin-top: 15px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .accession-tag {
        font-size: 12px;
        color: #64748B;
        background: #F1F5F9;
        padding: 3px 8px;
        border-radius: 5px;
        font-family: monospace;
    }

    /* Technical Specs Box */
    .specs-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        line-height: 1.6;
        color: #334155;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Technical Specs Box (KEEP THIS AS IS) */
    .specs-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        line-height: 1.6;
        color: #334155;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* ADD THIS RIGHT BELOW: Force Input Fields and Dropdown to Pure White */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }

    input {
        color: #0F172A !important;
        background-color: transparent !important;
    }

    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    }
    
    /* Diagnostic Card */
    .diagnosis-card {
        background: #F8FAFC;
        border-left: 4px solid #1E3A8A;
        border-radius: 0 8px 8px 0;
        padding: 18px 20px;
        margin-top: 15px;
        margin-bottom: 20px;
        font-size: 14px;
        line-height: 1.8;
    }

    /* Disclaimer */
    .disclaimer {
        font-size: 12px;
        color: #94A3B8;
        text-align: center;
        line-height: 1.6;
        margin-top: 35px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL LOADING & PREPROCESSING PIPELINE
# -----------------------------------------------------------------------------
MODEL_PATH = "pneumonia_resnet18_recall_optimized.pth"
THRESHOLD = 0.52

@st.cache_resource
def load_pytorch_model():
    """Loads ResNet-18 weights into memory once."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.error(f"❌ Model file '{MODEL_PATH}' was not found in the current working directory.")
        return None, device
    except Exception as e:
        st.error(f"❌ Error loading model state_dict: {e}")
        return None, device

    model.to(device)
    model.eval()
    return model, device

model, device = load_pytorch_model()

# Exact dataset pipeline (Resize 256 -> CenterCrop 224 -> Normalize 0.5)
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -----------------------------------------------------------------------------
# 3. TOP HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>PNEUMONIA DETECTION & DIAGNOSTIC SYSTEM</h1>
    <p>AI-Assisted Chest Radiograph Analysis • PyTorch ResNet-18</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. STEP 1: PATIENT REGISTRATION (2-COLUMN GRID)
# -----------------------------------------------------------------------------
auto_accession = f"ACC-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"

st.markdown(f"""
<div class="section-title">
    <span>1. Patient Information</span>
    <span class="accession-tag">{auto_accession}</span>
</div>
""", unsafe_allow_html=True)

# Row 1: Name and ID
col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input("Patient Full Name *", placeholder="Enter patient name")
with col2:
    patient_id = st.text_input("Patient ID / UHID *", placeholder="e.g., UHID-10023")

# Row 2: Age and Gender
col3, col4 = st.columns(2)
with col3:
    patient_age = st.text_input("Patient Age", placeholder="e.g., 45")
with col4:
    gender = st.selectbox("Gender", ["Select Gender", "Male", "Female", "Other"])

st.write("")

# -----------------------------------------------------------------------------
# 5. STEP 2: RADIOGRAPH ACQUISITION & UPLOAD
# -----------------------------------------------------------------------------
st.markdown("""
<div class="section-title">
    <span>2. Radiograph Acquisition</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Patient Chest X-Ray",
    type=["jpg", "jpeg", "png"],
    help="Upload frontal chest radiograph (AP/PA projection)"
)

# Display scan preview & technical specs if file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col_img, col_info = st.columns([1, 1.2])
    with col_img:
        st.image(image, caption="Uploaded Radiograph", use_container_width=True)
    with col_info:
        st.markdown(f"""
        <div class="specs-card">
            <strong>Original Dimensions:</strong> {image.size[0]} × {image.size[1]} px<br>
            <strong>Preprocess Pipeline:</strong> Resize (256×256) ➔ CenterCrop (224×224) ➔ Normalize (0.5)<br>
            <strong>Inference Hardware:</strong> {'Apple Silicon MPS' if device.type == 'mps' else 'CPU'}<br>
            <strong>Status:</strong> Ready for analysis
        </div>
        """, unsafe_allow_html=True)

# Run Analysis Button
analyze_btn = st.button("🔍 Run AI Analysis", use_container_width=True, type="primary")

# -----------------------------------------------------------------------------
# 6. STEP 3: DIAGNOSTIC EVALUATION (HIDDEN UNTIL BUTTON PRESSED WITH IMAGE)
# -----------------------------------------------------------------------------
if analyze_btn:
    if uploaded_file is None:
        st.warning("⚠️ Please upload a chest X-ray image before running analysis.")
    elif model is None:
        st.error("Model is not loaded. Please verify the .pth file path.")
    else:
        with st.spinner("Analyzing radiograph features..."):
            # Prepare tensor & run inference
            input_tensor = test_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(input_tensor)
                probability = torch.sigmoid(logit).item()

            is_pneumonia = probability >= THRESHOLD
            st.session_state["analyzed"] = True
            st.session_state["probability"] = probability
            st.session_state["is_pneumonia"] = is_pneumonia

# Display Diagnostic Evaluation if analysis has run
if st.session_state.get("analyzed", False) and uploaded_file is not None:
    prob = st.session_state["probability"]
    is_positive = st.session_state["is_pneumonia"]

    st.markdown("""
    <div class="section-title">
        <span>3. Diagnostic Evaluation</span>
    </div>
    """, unsafe_allow_html=True)

    # Format result text and color
    if is_positive:
        result_title = "Pneumonia Positive"
        result_color = "#991B1B"  # Crimson
    else:
        result_title = "Pneumonia Negative (Normal)"
        result_color = "#166534"  # Emerald green

    display_name = patient_name if patient_name.strip() else "Unspecified"
    display_id = patient_id if patient_id.strip() else "Unspecified"
    display_age = patient_age if patient_age.strip() else "N/A"
    display_gender = gender if gender != "Select Gender" else "N/A"

    st.markdown(f"""
    <div class="diagnosis-card">
        <div style="font-size: 18px; font-weight: 700; color: {result_color}; margin-bottom: 10px;">
            Diagnosis: {result_title}
        </div>
        <div><strong>Patient:</strong> {display_name} (ID: {display_id}) | <strong>Age:</strong> {display_age} | <strong>Gender:</strong> {display_gender}</div>
        <div style="margin-top: 4px;">
            <strong>Evaluation:</strong> Model probability is <strong>{prob:.3f}</strong> 
            ({'exceeds' if is_positive else 'is below'} the {THRESHOLD:.2f} recall-optimized cutoff).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action Buttons: Download Report & Reset
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        # Prepare text report
        summary_text = (
            f"CLINICAL RADIOGRAPH DIAGNOSTIC SUMMARY\n"
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Accession: {auto_accession}\n"
            f"---------------------------------------------------\n"
            f"Patient Name: {display_name}\n"
            f"Patient ID:   {display_id}\n"
            f"Age:          {display_age}\n"
            f"Gender:       {display_gender}\n"
            f"---------------------------------------------------\n"
            f"AI Assessment: {result_title}\n"
            f"Model Score:   {prob:.4f} (Threshold: {THRESHOLD})\n"
            f"Architecture:  ResNet-18 (Recall-Optimized)\n"
            f"---------------------------------------------------\n"
            f"Note: Decision-support tool for academic validation.\n"
        )
        st.download_button(
            label="📥 Download Diagnostic Summary (.txt)",
            data=summary_text,
            file_name=f"Diagnosis_{display_id}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with btn_col2:
        if st.button("🔄 Start New Diagnostic Session", use_container_width=True):
            st.session_state["analyzed"] = False
            st.rerun()

# -----------------------------------------------------------------------------
# 7. CLINICAL DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Advisory:</strong> This deep learning interface is developed for clinical decision support 
    and internship demonstration only. Final diagnosis must always be confirmed by a certified radiologist.
</div>
""", unsafe_allow_html=True)