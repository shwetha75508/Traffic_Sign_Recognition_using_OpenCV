import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)


# Class ID → Class name

class_names = {
    0: "Speed limit (20 km/h)",
    1: "Speed limit (30 km/h)",
    2: "Speed limit (50 km/h)",
    3: "Speed limit (60 km/h)",
    4: "Speed limit (70 km/h)",
    5: "Speed limit (80 km/h)",
    6: "End of speed limit (80 km/h)",
    7: "Speed limit (100 km/h)",
    8: "Speed limit (120 km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals ahead",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing for vehicles over 3.5 tons"
}


# Load the trained KNN model

@st.cache_resource
def load_model():
    return joblib.load("traffic_sign_knn_model.pkl")


model = load_model()


# HOG Feature extraction

def extract_hog(image):

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to the same size used during training
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

    # Extract HOG features
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False
    )

    return hog_features.reshape(1, -1)


# Prediction

def predict(image):

    # Extract HOG features
    features = extract_hog(image)

    # Predict class
    class_id = int(model.predict(features)[0])

    # Get probability for each class
    probabilities = model.predict_proba(features)[0]

    # probability of predicted class
    confidence = probabilities[class_id] * 100

    # Get class name
    class_name = class_names.get(class_id, "Unknown Traffic Sign")

    return class_id, class_name, confidence


# Application

st.title("🚦 Traffic Sign Recognition")

st.write(
    "Recognize traffic signs from images using **OpenCV**, **HOG feature** and **KNN**."
)

st.divider()


# Image Upload

st.subheader("📷 Upload Traffic Sign Image")

uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# Image Recognition

if uploaded_image is not None:

    image = Image.open( uploaded_image).convert("RGB")

    st.image(image, caption="Uploaded Image",width=350)

    # Convert RGB → BGR
    image_array = np.array(image)

    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Predict
    class_id, class_name, confidence = predict(image_bgr)


    # Display prediction

    st.success(
        f"🚦 Predicted Traffic Sign: {class_name}"
    )

    # Display Class ID & Confidence
    col1, col2 = st.columns(2)

    with col1:
        st.metric("🎯 Class ID", class_id)

    with col2:
        st.metric("📈 Confidence Score", f"{confidence:.2f}%")


# Model Information

st.divider()

st.subheader("📊 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Traffic Sign Classes", "43")

    st.metric("Image Size", "32 * 32")

with col2:
    st.metric("HOG Features", "324")

    st.metric("Classifier", "KNN") 


# Model Details

with st.expander("ℹ️ About the Model"):
    st.markdown(
        """
    **Traffic Sign Recognition Model**

    This application uses **OpenCV**, **HOG feature extraction**,
    and **K-Nearest Neighbors (KNN)** to classify traffic signs.

    **Preprocessing**
    - Images are converted to grayscale
    - Images are resized to **32 × 32 pixels**

    **Feature Extraction**
    - Histogram of Oriented Gradients (HOG)
    - 9 orientations
    - 8 × 8 pixels per cell
    - 2 × 2 cells per block
    - L2-Hys normalization
    - **324 features per image**

    **Classification**
    - K-Nearest Neighbors (KNN)
    - **43 traffic sign classes**

    **Model Limitation**

    The model is trained to classify images into the 43 traffic sign classes present in the dataset. 

    Images outside these 43 classes, or images from the same 43 classes 
    that differ significantly from the training data in terms of lighting, background, cropping, scale, angle, or image quality, may be classified incorrectly.
    
    The predicted class and confidence score should therefore be
    interpreted within the scope of the training dataset.
    """
    )

        
# Footer

st.divider()
st.caption("Traffic Sign Recognition using OpenCV, HOG and KNN")    
