from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import base64
from keras_preprocessing import image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load the pre-trained model (ensure your model is in the same directory or provide the full path)
model = tf.keras.models.load_model(r'C:\Users\91897\Dropbox\PC\Downloads\my_flask_app\Densenet201.h5')

# Function to preprocess the image for the model
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize the image to the model's input size
    img_array = np.array(img) / 255.0  # Normalize the image to the range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img = Image.open(file.stream)  # Open the image from the uploaded file
        img_array = preprocess_image(img)  # Preprocess the image

        # Predict the class
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)  # Get the class with the highest probability

        # For example, we can map the prediction index to a class label
        # Assuming you have a list of class names like ["Grade 1", "Grade 2", ...]
        class_labels = ["Normal", "Grade 1", "Grade 2", "Grade 3"]  # Example labels
        predicted_class = class_labels[class_idx[0]]

        # Convert the image to base64 so it can be displayed in the browser
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'prediction': predicted_class,
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
