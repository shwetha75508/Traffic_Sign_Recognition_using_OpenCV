# 🚦 Traffic Sign Recognition using Computer Vision and Machine Learning

---

## 📌 Project Overview
This project is a Computer Vision and Machine Learning application that recognizes **43 different traffic sign classes** from uploaded images.

The system uses **OpenCV** for image preprocessing, **Histogram of Oriented Gradients (HOG)** for feature extraction, and **K-Nearest Neighbors (KNN)** for classification. The application predicts the traffic sign class and displays the corresponding **Class ID, predicted sign name, and confidence score**.

The model was trained and evaluated using a traffic sign dataset containing **39,262 images across 43 classes**. Images were resized to **32 × 32 pixels**, converted to grayscale, and transformed into **324 HOG features** before classification.

The trained model is deployed through a **Streamlit application**, providing a simple and interactive interface for traffic sign image recognition.

---

🎯 Objectives
- Recognize traffic signs from uploaded images.
- Apply image preprocessing using OpenCV.
- Extract shape and edge information using HOG.
- Train and evaluate multiple machine learning classification models.
- Use KNN as the final classification model.
- Display the predicted class, class ID, and confidence score.
- Deploy the trained model through Streamlit.

---

📂 Dataset

The project uses a GTSRB - German Traffic Sign Recognition Benchmark dataset containing:

- 39,262 images
- 43 traffic sign classes
- Images organized according to their class IDs.
- Images have different original resolutions and sizes.

## 🛠️ Technologies Used

* Python
* OpenCV
* Scikit-learn
* Scikit-image (HOG)
* NumPy
* Pandas
* Joblib
* Streamlit


