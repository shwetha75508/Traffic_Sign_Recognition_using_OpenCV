# 🚦 Traffic Sign Recognition using Computer Vision and Machine Learning


## 📌 Project Overview
This project is a Computer Vision and Machine Learning application that recognizes **43 different traffic sign classes** from uploaded images.

The system uses **OpenCV** for image preprocessing, **Histogram of Oriented Gradients (HOG)** for feature extraction, and **K-Nearest Neighbors (KNN)** for classification. The application predicts the traffic sign class and displays the corresponding **Class ID, predicted sign name, and confidence score**.

The model was trained and evaluated using a traffic sign dataset containing **39,262 images across 43 classes**. Images were resized to **32 × 32 pixels**, converted to grayscale, and transformed into **324 HOG features** before classification.

The trained model is deployed through a **Streamlit application**, providing a simple and interactive interface for traffic sign image recognition.

---

## 🎯 Objectives
- Recognize traffic signs from uploaded images.
- Apply image preprocessing using OpenCV.
- Extract shape and edge information using HOG.
- Train and evaluate multiple machine learning classification models.
- Use KNN as the final classification model.
- Display the predicted class, class ID, and confidence score.
- Deploy the trained model through Streamlit.

---

## 📂 Dataset

The project uses a GTSRB - German Traffic Sign Recognition Benchmark dataset containing:

- 39,262 images
- 43 traffic sign classes
- Images organized according to their class IDs.
- Images have different original resolutions and sizes.

---  

## 🛠️ Technologies Used

- Python
- OpenCV
- Scikit-learn
- Scikit-image (HOG)
- NumPy
- Pandas
- Joblib
- Streamlit

---

## 🔄 Machine Learning Workflow

```text
Dataset Loading
      ↓
Data Validation
      ↓
Image Preprocessing
      ↓
Image Resizing (32 × 32)
      ↓
Grayscale Conversion
      ↓
HOG Feature Extraction (324 Features)
      ↓
Train-Test Split (80:20)
      ↓
Model Training
      ↓
Model Evaluation
      ↓
KNN Classification
      ↓
Model Serialization (Joblib)
      ↓
Streamlit Deployment
```


---

## 🤖 Machine Learning Models

Several classification algorithms were evaluated during the project, including:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Naive Bayes

The **K-Nearest Neighbors (KNN)** classifier was selected as the final model and deployed in the Streamlit application.

### Final Model

- **Algorithm:** K-Nearest Neighbors (KNN)
- **Number of Classes:** 43
- **Input:** 32 × 32 grayscale image
- **HOG Features:** 324
- **Test Accuracy:** Approximately 97%

The reported accuracy is based on the **held-out test set from the same dataset distribution** used during model development.

---

## 🌐 Streamlit Application

The application allows users to:

1. Upload a traffic sign image.
2. Process the image using the same preprocessing pipeline used during training.
3. Extract HOG features.
4. Predict the traffic sign class.
5. Display:
   - Class ID
   - Predicted Traffic Sign
   - Confidence Score
6. View information about the model and its limitations.

---

## ⚠️ Model Limitations

The model is trained to classify images into the **43 traffic sign classes present in the dataset**.

It does not currently include an **Unknown / Not a Traffic Sign** detection class.

Images outside these 43 classes, or images from the same 43 classes that differ significantly from the training data in terms of **lighting, background, cropping, scale, viewing angle, or image quality**, may be classified incorrectly.

The confidence score represents the model's classification confidence and should not be interpreted as a guarantee that the prediction is correct.

---

## 🚀 Future Enhancements

- Improve model generalization using more diverse real-world images.
- Add data augmentation to increase training-data variability. 
- Implement CNN-based deep learning models.
- Add Unknown / Out-of-Distribution detection.
- Improve recognition under different lighting and viewing conditions.
- Implement object detection for traffic signs in complete road scenes.
- Develop real-time traffic sign recognition.

---

## 👨‍💻 Author
Shweta Rani
- LinkedIn: 
