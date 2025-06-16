# -------------------- Streamlit UI Setup --------------------
import streamlit as st
from PIL import Image
import numpy as np
import time
from keras.applications import *
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

st.set_page_config(page_title="🧠 AI Model Hub", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 3rem;
        font-weight: bold;
        font-family: 'Segoe UI', sans-serif;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .stSidebar > div:first-child {
        background-color: #d0f0c0;
        border-radius: 10px;
        padding: 1rem;
    }
    .custom-footer {
        background: linear-gradient(to right, #2E8B57, #1E5631);
        color: white;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 16px;
        margin-top: 40px;
        font-weight: 500;
    }
    .social-icons a {
        margin: 0 10px;
        color: white;
        text-decoration: none;
        font-size: 20px;
    }
    </style>
    <h1 class="main-header">🧠 AI Model Hub: Vision + Augmentation + Analysis</h1>
    <div class="sub-header">Explore cutting-edge models with ease – Augment, Classify, and Analyze Images 🚀</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar Inputs --------------------
st.sidebar.title("⚙️ Model & Options")
task = st.sidebar.radio("🧪 Choose Task", ["Image Augmentation", "Image Classification"])
model_name = st.sidebar.selectbox("🤖 Select Model", [
    "ResNet50", "ResNet50V2", "VGG16", "VGG19", "Xception", "InceptionV3",
    "MobileNetV2", "DenseNet121", "NASNetMobile", "NASNetLarge", "EfficientNetV2B0"
]) if task == "Image Classification" else None

uploaded_file = st.sidebar.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# -------------------- Model Loader --------------------
@st.cache_resource
def load_model(model_name):
    model_dict = {
        "ResNet50": ResNet50,
        "ResNet50V2": ResNet50V2,
        "VGG16": VGG16,
        "VGG19": VGG19,
        "Xception": Xception,
        "InceptionV3": InceptionV3,
        "MobileNetV2": MobileNetV2,
        "DenseNet121": DenseNet121,
        "NASNetMobile": NASNetMobile,
        "NASNetLarge": NASNetLarge,
        "EfficientNetV2B0": EfficientNetV2B0,
    }
    return model_dict[model_name](weights='imagenet')

preprocessors = {
    "ResNet50": resnet50.preprocess_input,
    "ResNet50V2": resnet_v2.preprocess_input,
    "VGG16": vgg16.preprocess_input,
    "VGG19": vgg19.preprocess_input,
    "Xception": xception.preprocess_input,
    "InceptionV3": inception_v3.preprocess_input,
    "MobileNetV2": mobilenet_v2.preprocess_input,
    "DenseNet121": densenet.preprocess_input,
    "NASNetMobile": nasnet.preprocess_input,
    "NASNetLarge": nasnet.preprocess_input,
    "EfficientNetV2B0": efficientnet_v2.preprocess_input,
}

input_sizes = {
    "Xception": (299, 299),
    "InceptionV3": (299, 299),
    "NASNetLarge": (331, 331)
}

# -------------------- Augmentation Function --------------------
def generate_augmented_images(image_pil):
    datagen = ImageDataGenerator(
        rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
    )
    x = img_to_array(image_pil)
    x = x.reshape((1,) + x.shape)
    images = [Image.fromarray(batch[0].astype("uint8")) for i, batch in zip(range(6), datagen.flow(x, batch_size=1))]
    return images

# -------------------- Inference Function --------------------
def classify_image(model, image_pil, model_name):
    size = input_sizes.get(model_name, (224, 224))
    image_pil = image_pil.crop(image_pil.getbbox())
    image_resized = image_pil.resize(size)
    x = img_to_array(image_resized)
    x = np.expand_dims(x, axis=0)
    preprocess_func = preprocessors[model_name]
    x = preprocess_func(x)

    start_time = time.time()
    preds = model.predict(x)
    end_time = time.time()

    inference_time_ms = (end_time - start_time) * 1000.0
    num_parameters = model.count_params()
    model_depth = len(model.layers)
    model_size_MB = num_parameters * 4 / (1024 ** 2)
    decoded_preds = decode_predictions(preds, top=5)[0]

    return decoded_preds, inference_time_ms, num_parameters, model_depth, model_size_MB

# -------------------- Main Interface --------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    if task == "Image Augmentation":
        st.subheader("🔁 Augmented Images")
        aug_images = generate_augmented_images(image)
        cols = st.columns(3)
        for i, img in enumerate(aug_images):
            with cols[i % 3]:
                st.image(img, caption=f"Aug #{i+1}", use_column_width=True)

    elif task == "Image Classification":
        st.subheader(f"📊 Classification using: {model_name}")
        model = load_model(model_name)
        preds, inference_time, num_params, depth, size_mb = classify_image(model, image, model_name)

        for i, (imagenetID, label, prob) in enumerate(preds):
            st.success(f"🔹 Top {i+1}: **{label}** ({prob*100:.2f}%)")

        with st.expander("🧠 Model Analysis"):
            st.markdown(f"""
            - 🕒 **Inference Time**: `{inference_time:.2f} ms`  
            - 📦 **Total Parameters**: `{num_params}`  
            - 🧱 **Model Depth**: `{depth} layers`  
            - 💾 **Model Size**: `{size_mb:.2f} MB`
            """)

# -------------------- Footer --------------------
st.markdown("""
    <div class="custom-footer">
        🌟 Created with ❤️ by Mahesh Babu 
    </div>
""", unsafe_allow_html=True)
