# app.py

# Import necessary modules:
# - Flask and its helpers for creating the web server and handling requests/responses.
# - numpy for numerical operations (especially for reshaping input features).
# - pickle for loading the serialized machine learning model.
# - logging and RotatingFileHandler for setting up logging to track API requests and errors.
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import logging
from logging.handlers import RotatingFileHandler

# Initialize the Flask application.
app = Flask(__name__)

# ------------------------------
# Logging Setup
# ------------------------------
# Configure logging so that all API activity is recorded.
# RotatingFileHandler is used to manage log file sizes, ensuring that logs are rotated after reaching 1MB.
handler = RotatingFileHandler('app.log', maxBytes=1_000_000, backupCount=3)
handler.setLevel(logging.INFO)  # Set the log level to INFO.
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')  # Define the log format.
handler.setFormatter(formatter)
app.logger.addHandler(handler)  # Attach the handler to the Flask app's logger.
app.logger.setLevel(logging.INFO)  # Ensure that the logger captures INFO level logs and above.

# ------------------------------
# Load the Trained Model
# ------------------------------
# Open and load the serialized model (house_price_model.pkl) using pickle.
# The model is loaded once at startup so that subsequent prediction requests can use it directly.
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)


# ------------------------------
# Define Routes
# ------------------------------

@app.route('/')
def home():
    """
    Render the home page which serves as a simple frontend UI.
    This UI is defined in the 'index.html' template located in the 'templates' directory.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction Endpoint: Accepts JSON input with house feature values and returns the predicted house price.

    Process:
    1. Extract JSON data from the POST request.
    2. Log the incoming request data for debugging and traceability.
    3. Convert the input data (dictionary) into a numpy array and reshape it as required by the model.
    4. Use the loaded model to make a prediction.
    5. Log the prediction result.
    6. Return the prediction in JSON format.

    Error Handling:
    - Any exceptions during data processing or prediction are caught.
    - The error is logged with exception details.
    - A JSON response with the error message is returned along with a 400 status code.
    """
    try:
        # Get JSON data from the request body
        data = request.get_json()
        app.logger.info(f"Received request data: {data}")

        # Convert the input data (assumed to be a dictionary of features) into a 2D numpy array
        input_features = np.array(list(data.values())).reshape(1, -1)

        # Use the loaded model to make a prediction; extract the first element from the returned array
        prediction = model.predict(input_features)[0]

        # Log successful prediction
        app.logger.info(f"Prediction successful: {prediction}")

        # Return the predicted price as JSON
        return jsonify({'predicted_price': prediction})
    except Exception as e:
        # Log the error with a full traceback for debugging purposes
        app.logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        # Return a JSON response with the error message and a 400 status code
        return jsonify({'error': str(e)}), 400


# ------------------------------
# Run the Application
# ------------------------------
if __name__ == '__main__':
    # Start the Flask app in debug mode.
    # Debug mode is useful during development, but should be disabled in production.
    app.run(debug=True)
