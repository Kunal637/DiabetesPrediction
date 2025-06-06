Diabetes Prediction App
This project provides a Streamlit-based web application to predict the likelihood of diabetes using a K-Nearest Neighbors (KNN) machine learning model. The model is trained on a dataset containing health metrics such as Pregnancies, Glucose, BMI, Age, and Diabetes Pedigree Function, achieving approximately 98.6% accuracy on the test set.
Table of Contents

Overview
Prerequisites
Installation
Dataset
Usage
Project Structure
Model Details
Limitations
Contributing
License

Overview
The application allows users to input health metrics and receive a prediction on whether they are likely to have diabetes, along with probability scores. The model was trained and optimized using scikit-learn, with preprocessing steps including outlier handling (winsorizing) and feature scaling. The Streamlit app provides a user-friendly interface for real-time predictions.
Prerequisites

Python 3.8 or higher
Git
A GitHub account (to clone the repository)
Visual Studio Code (optional, for development)
Internet access (to download the dataset and dependencies)

Installation

Clone the Repository:
git clone https://github.com/your-username/DiabetesPrediction.git
cd DiabetesPrediction

Replace your-username with your GitHub username.

Set Up a Virtual Environment:
python -m venv venv

Activate the virtual environment:

Windows: venv\Scripts\activate
macOS/Linux: source venv/bin/activate


Install Dependencies:
pip install -r requirements.txt



Dataset
The model requires the diabetes.csv dataset, which is not included in the repository due to potential size or sensitivity constraints. You can obtain it from:

Kaggle: Pima Indians Diabetes Database (ensure it includes columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome).
Alternatively, contact the repository owner for a synthetic dataset or specific download instructions.

Place diabetes.csv in the project root directory before running the training script.
Usage

Train the Model:Run the training script to preprocess the dataset, train the KNN model, and save the model and scaler:
python train_model.py

This generates best_knn_model.pkl and scaler.pkl in the project directory.

Run the Streamlit App:Launch the Streamlit application:
streamlit run diabetes_prediction_app.py

The app will open in your default web browser.

Interact with the App:

Enter values for Pregnancies, Glucose, BMI, Age, and Diabetes Pedigree Function in the input fields.
Click the “Predict” button to view the prediction (Positive/Negative) and probability scores.
Example input (based on typical dataset ranges):
Pregnancies: 0–17
Glucose: 50–200 mg/dL
BMI: 17.5–50 kg/m²
Age: 21–81 years
Diabetes Pedigree Function: 0.078–2.42





Project Structure

train_model.py: Script to load diabetes.csv, preprocess data, train and optimize the KNN model, and save the model and scaler.
diabetes_prediction_app.py: Streamlit application for user input and diabetes prediction.
requirements.txt: List of Python dependencies.
.gitignore: Excludes virtual environments, pickle files (best_knn_model.pkl, scaler.pkl), and diabetes.csv.
README.md: This documentation file.

Generated files (not in repository):

best_knn_model.pkl: Trained KNN model.
scaler.pkl: Fitted StandardScaler for preprocessing.
diabetes.csv: Dataset (must be obtained separately).

Model Details

Algorithm: K-Nearest Neighbors (KNN)
Hyperparameters: n_neighbors=5, weights='distance', metric='euclidean' (optimized via GridSearchCV)
Features: Pregnancies, Glucose, BMI, Age, Diabetes Pedigree Function
Preprocessing:
Outliers handled using winsorizing (5% limits).
Features scaled using StandardScaler.


Performance: ~98.6% accuracy on the test set (15% of data, stratified split).
Training: Uses

