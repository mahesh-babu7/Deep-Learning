import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="FaceFeel AI", page_icon="🧠", layout="wide")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("emotion_model.h5")

model = load_model()

# ------------------- Theme Setup -------------------
theme = st.sidebar.selectbox("🎨 Choose Theme", ["Aqua Glow", "Lavender Light", "Sunset Pulse"])

theme_colors = {
    "Aqua Glow": {"bg": "#ecfaff", "text": "#0abde3", "bar": "#00cec9", "shadow": "0 0 20px #00f2ff"},
    "Lavender Light": {"bg": "#f9f5ff", "text": "#9b59b6", "bar": "#be93f5", "shadow": "0 0 15px #e0c3fc"},
    "Sunset Pulse": {"bg": "#fff6f0", "text": "#e17055", "bar": "#fab1a0", "shadow": "0 0 12px #ff7675"},
}

colors = theme_colors[theme]

# ------------------- CSS -------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rubik:wght@600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Segoe UI', sans-serif;
    background-color: {colors['bg']};
}}

section[data-testid="stFileUploader"] label {{
    font-family: 'Segoe UI', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    color: #444 !important;
}}

.main-header {{
    font-family: 'Rubik', sans-serif;
    font-size: 3rem;
    text-align: center;
    font-weight: 600;
    color: transparent;
    background: linear-gradient(90deg, {colors['text']}, #555);
    -webkit-background-clip: text;
    background-clip: text;
    margin-bottom: 30px;
    position: relative;
    animation: fadeIn 1s ease-in-out;
}}

.main-header::after {{
    content: '';
    display: block;
    width: 60px;
    height: 4px;
    background: {colors['bar']};
    margin: 10px auto 0 auto;
    border-radius: 5px;
    animation: slide 2s ease-in-out infinite;
}}

.emoji {{
    font-size: 4rem;
    text-align: center;
    margin-top: 100px;
    animation: pulse 1s ease-in-out infinite alternate;
}}

.prediction-text {{
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
    margin-left: 30px;
    color: #111;
}}

.confidence-bar {{
    background-color: #ddd;
    border-radius: 30px;
    overflow: hidden;
    height: 25px;
    margin: 10px auto;
    width: 80%;
    margin-left: 70px;
    box-shadow: inset 0 0 5px rgba(0,0,0,0.2);
}}

.confidence-fill {{
    height: 100%;
    background: {colors['bar']};
    text-align: center;
    color: #fff;
    font-weight: bold;
    line-height: 25px;
    transition: width 0.8s ease;
}}

.footer {{
    text-align: center;
    font-size: 0.9rem;
    color: #555;
    margin-top: 3rem;
    border-top: 1px solid #ccc;
    padding-top: 1rem;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(-20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes slide {{
    0%, 100% {{ transform: translateX(0); }}
    50% {{ transform: translateX(20px); }}
}}
</style>
""", unsafe_allow_html=True)

# ------------------- Sidebar -------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4149/4149684.png", width=80)
st.sidebar.title("FaceFeel AI – Instant Emotion Reader")
st.sidebar.markdown(f"""
AI-powered facial emotion detection.

### ⚙️ Features
- Detects **Happy 😀** or **Sad 😢**
- Emoji, Prediction & Confidence
- 3 Cool Themes

### 🧠 Built With
- TensorFlow CNN  
- Streamlit  
- Python 🐍

📦 Version: `v2.2.0`  
📬 [Contact Us](mailto:support@moodmirror.ai)
""")

# ------------------- Header -------------------
st.markdown(f'<div class="main-header">FaceFeel AI – Instant Emotion Reader</div>', unsafe_allow_html=True)

# ------------------- Main Row Layout -------------------
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("📤 Upload")
    uploaded_file = st.file_uploader("Upload your face image", type=["jpg", "jpeg", "png"])

with col2:
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Image", width=350)

        # Resize for model prediction only (not display)
        img_resized = img.resize((200, 200))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]
        label = "Sad 😢" if prediction > 0.5 else "Happy 😀"
        emoji = "😢" if prediction > 0.5 else "😀"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        confidence_percent = int(confidence * 100)

        # Share data via session
        st.session_state['label'] = label
        st.session_state['emoji'] = emoji
        st.session_state['confidence_percent'] = confidence_percent

with col3:
    if uploaded_file and 'label' in st.session_state:
        st.markdown(f'<div class="emoji">{st.session_state["emoji"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-text">Prediction: {st.session_state["label"]}</div>', unsafe_allow_html=True)
        st.markdown(f'''
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {st.session_state["confidence_percent"]}%; ">
                    {st.session_state["confidence_percent"]}%
                </div>
            </div>
        ''', unsafe_allow_html=True)

# ------------------- Footer -------------------
st.markdown(f"""
<div class="footer">
    © 2025 FaceFeel AI · Theme: <strong>{theme}</strong> · Built with ❤️ and 🤖
</div>
""", unsafe_allow_html=True)
