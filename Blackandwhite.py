import streamlit as st
import numpy as np
import cv2
from cv2 import dnn
from PIL import Image
import tempfile

# Load the pre-trained model
proto_file = 'colorization_deploy_v2.prototxt'
model_file = 'colorization_release_v2.caffemodel'
hull_pts = 'pts_in_hull.npy'

net = dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)

# Load cluster centers into the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Streamlit App UI
st.set_page_config(page_title="NeuralTint", page_icon="🎨", layout="centered")
st.markdown("""
    <style>
    .stApp { background-color: #f5f5f5; }
    .title { 
        text-align: center; 
        font-size: 34px; /* Slightly larger for better readability */
        font-weight: bold; 
        color: #111; /* Darker text for strong contrast */
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3); /* More pronounced shadow */
    }
    .subtitle { 
        text-align: center; 
        font-size: 18px; 
        color: #222; /* Darker than #444 to improve contrast */
    }
    .uploaded-image { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 class='title'>NeuralTint - Restore Colors Instantly</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Breathe color into your grayscale photos effortlessly!</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "png", "jpeg"], help="Supported formats: JPG, PNG, JPEG")

if uploaded_file is not None:
    # Read and process the image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    # Process the L channel
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    # Predict the ab channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))
    
    # Combine channels
    L = cv2.split(lab_img)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Grayscale Image", width=300, use_container_width=True)
    with col2:
        st.image(Image.fromarray(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)), caption="Colorized Image", width=300, use_container_width=True)
    
    # Allow downloading the colorized image
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)).save(temp_file.name)
    st.download_button("Download Colorized Image", temp_file.name, file_name="colorized.png", mime="image/png")
