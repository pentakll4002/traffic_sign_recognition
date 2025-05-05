# 🚦 Traffic Sign Recognition using Machine Learning and Deep Learning

A machine learning and deep learning project for classifying traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

---

## 📚 Libraries Used

- [NumPy](https://numpy.org/): For numerical computations  
- [Pandas](https://pandas.pydata.org/): For handling tabular data  
- [Seaborn](https://seaborn.pydata.org/): For data visualization  
- [Matplotlib](https://matplotlib.org/): For plotting graphs and images  
- [scikit-learn](https://scikit-learn.org/): For preprocessing, model evaluation, and traditional ML models  
- [TensorFlow / Keras](https://www.tensorflow.org/): For building and training deep learning models

---

## 📁 Project Structure

traffic_sign_recognition/
│
├── data/ # Dataset (GTSRB)
│ ├── Train/
│ └── Test/
│
├── notebooks/ # Jupyter notebooks for exploration and training
│ └── traffic_sign_recognition.ipynb
│
├── models/ # Saved models (HDF5 or .pkl)
│
├── utils/ # Utility functions (preprocessing, plotting, etc.)
│
├── README.md
└── requirements.txt

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/traffic_sign_recognition.git
cd traffic_sign_recognition
```

### 2. Install Dependencies
 
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
You can download the GTSRB dataset from here and extract it into the data/ folder.

### 4. Run Jupyter Notebook

```bash
jupyter notebook notebooks/traffic_sign_recognition.ipynb
```

##  📊 Workflow Overview
1. Data Loading & Preprocessing

- Load and resize images

- Normalize pixel values

- One-hot encode labels

2. EDA (Exploratory Data Analysis)

- Class distribution

- Sample visualizations using matplotlib and seaborn

3. Model Building

- Classical ML (Random Forest, SVM using scikit-learn)

- CNN with TensorFlow/Keras

4. Model Evaluation

- Accuracy, Confusion Matrix

- Visualization of predictions

5. Saving and Loading Models

- Save trained models to disk (.h5 or .pkl)

## 📈 Sample Results
- CNN Accuracy: ~98% on test set

- SVM Accuracy: ~92% (after feature extraction)

## ✅ To-Do
- Add real-time traffic sign detection using OpenCV

- Optimize CNN using Data Augmentation

- Export model to TFLite for mobile deployment

## 📄 License
This project is licensed under the MIT License.
