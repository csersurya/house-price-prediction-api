# House Price Prediction API

This project builds and deploys a machine learning model to predict house prices. The solution covers data preprocessing, model training with hyperparameter tuning, and deployment as a REST API using Flask. Additionally, logging, error handling, and a simple frontend UI are included.

## Overview

- **Data Preprocessing & Feature Engineering:**  
  The dataset (California Housing Dataset) was loaded and analyzed using exploratory data analysis (EDA). Missing values were handled via imputation, numerical features were scaled, and relevant features were selected based on correlation analysis.

- **Model Selection & Optimization:**  
  A Random Forest Regressor was chosen as the baseline model. The model was evaluated using RMSE, MAE, and R² metrics. Hyperparameter tuning was performed using GridSearchCV to optimize model performance.

- **Deployment Strategy:**  
  The trained model is saved using Pickle and served via a Flask API. The API exposes a `/predict` endpoint that accepts JSON input and returns the predicted price. Logging and error handling are integrated into the Flask application for better monitoring. A simple frontend UI is provided for quick testing.

## Project Structure

```
├── app.py                  # Flask application with logging, error handling, and API endpoints
├── house_price_model.pkl   # Serialized model saved after training & tuning
├── requirements.txt        # List of required Python packages
├── templates/
│   └── index.html          # Simple frontend UI for testing predictions
└── README.md               # This file
```

## Steps Taken

### 1. Data Preprocessing and Feature Engineering

- **Data Loading & EDA:**  
  The California Housing Dataset was loaded, and initial data exploration was performed using pandas to check for missing values and data types. A correlation heatmap (using Seaborn) was generated to visualize the relationships between features and the target variable.

- **Handling Missing Values:**  
  Missing values were handled with a median imputation strategy to maintain the integrity of the dataset.

- **Scaling and Feature Selection:**  
  Numerical features were scaled using `StandardScaler` from Scikit-learn to standardize feature distributions. Irrelevant or low-correlation features were dropped to improve model performance.

### 2. Model Training & Optimization

- **Data Splitting:**  
  The dataset was split into training and testing sets using an 80/20 ratio.

- **Model Training:**  
  A Random Forest Regressor was initially trained on the training data.

- **Model Evaluation:**  
  Predictions were evaluated using RMSE, MAE, and R² scores. These metrics helped assess the baseline performance.

- **Hyperparameter Tuning:**  
  GridSearchCV was used to fine-tune parameters such as `n_estimators`, `max_depth`, and `min_samples_split`. The best model was selected based on cross-validation performance, and its metrics were compared to the baseline.

- **Model Persistence:**  
  The best performing model was saved using Pickle for later use in the API.

### 3. Deployment Strategy & API Usage

- **Flask API:**  
  A Flask application (`app.py`) loads the serialized model at startup and provides a `/predict` endpoint. The endpoint accepts JSON data containing the house features, makes a prediction using the loaded model, and returns the predicted price.

- **Logging & Error Handling:**  
  The API uses Python’s `logging` module to log incoming requests and errors to a rotating log file (`app.log`). This facilitates debugging and production monitoring.

- **Frontend UI:**  
  A simple HTML page (`templates/index.html`) is provided to allow users to input house features, submit prediction requests, and view the results directly from the browser.

## How to Run the Project

### Prerequisites

- Python 3.9 or later
- Required packages listed in `requirements.txt`

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/house-price-prediction-api.git
   cd house-price-prediction-api
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask Application:**
   ```bash
   python app.py
   ```
   The API will start on `http://127.0.0.1:5000`.

2. **Access the Frontend UI:**
   Open your web browser and navigate to `http://127.0.0.1:5000/` to access the simple UI for making predictions.

3. **Using CURL or Postman:**
   To test the `/predict` endpoint directly, you can use CURL:
   ```bash
   curl -X POST -H "Content-Type: application/json" \
       -d '{"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.9841, "AveBedrms": 1.0238, "Population": 322.0, "AveOccup": 2.5556, "Latitude": 37.88, "Longitude": -122.23}' \
       http://127.0.0.1:5000/predict
   ```

3. **Access the API:**
   Once the container is running, the API will be available at `http://localhost:5000/`.

## Conclusion

This project demonstrates the end-to-end process of building a machine learning model for house price prediction, including data preprocessing, model training, hyperparameter tuning, and deploying the model as a REST API with logging and error handling. The project is modular, well-documented, and ready for further enhancements or integration into larger systems.

For any questions or contributions, please feel free to open an issue or submit a pull request.
