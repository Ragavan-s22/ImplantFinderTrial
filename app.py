from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MODEL_PATH'] = 'autoencoder_model.h5'

# Load the trained model with error handling
try:
    autoencoder = load_model(app.config['MODEL_PATH'])
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Function to process and predict the uploaded image
def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    reconstructed_img = autoencoder.predict(img)[0]
    diff = np.abs(img[0].squeeze() - reconstructed_img.squeeze())

    return reconstructed_img.squeeze(), diff

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'xray-upload' not in request.files:
        return redirect(request.url)
    file = request.files['xray-upload']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            reconstructed_img, diff = process_image(filepath)
        except Exception as e:
            return f"Error processing image: {e}", 500

        # Save the result images
        result_path = app.config['RESULT_FOLDER']
        plt.imsave(os.path.join(result_path, f'reconstructed_{filename}'), reconstructed_img, cmap='gray')
        plt.imsave(os.path.join(result_path, f'diff_{filename}'), diff, cmap='hot')

        return redirect(url_for('uploaded_file', filename=filename))
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['RESULT_FOLDER'], f'diff_{filename}')
    except FileNotFoundError:
        abort(404)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(debug=True)

