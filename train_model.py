import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mstats
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path='diabetes.csv'):
    """Load the diabetes dataset."""
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print("Error: 'diabetes.csv' not found.")
        return None

def preprocess_data(df):
    """Preprocess the dataset: handle outliers, split data, and scale features."""
    # Define relevant features and target
    features = ['Pregnancies', 'Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
    target = 'Outcome'

    # Handle outliers using winsorizing (as per notebook)
    for col in features:
        df[col] = mstats.winsorize(df[col], limits=[0.05, 0.05])

    # Split features and target
    X = df[features]
    y = df[target]

    # Split data into train (70%), validation (15%), test (15%) with stratification
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )  # 0.1765 ≈ 15/(100-15)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def train_and_optimize_model(X_train, y_train, X_val, y_val):
    """Train and optimize KNN model using GridSearchCV."""
    # Define parameter grid for KNN (based on notebook results)
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Initialize and fit GridSearchCV
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_val, y_val)  # Optimize on validation set

    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Evaluate on validation set
    from sklearn.metrics import accuracy_score
    val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    return best_model

def save_model_and_scaler(model, scaler, model_path='best_knn_model.pkl', scaler_path='scaler.pkl'):
    """Save the model and scaler to pickle files."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Model saved to {model_path}, scaler saved to {scaler_path}")

def main():
    # Load data
    df = load_data()
    if df is None:
        return

    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(df)

    # Train and optimize model
    best_model = train_and_optimize_model(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    from sklearn.metrics import accuracy_score
    test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save model and scaler
    save_model_and_scaler(best_model, scaler)

if __name__ == "__main__":
    main()