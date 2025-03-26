from flask import Flask, request, jsonify, render_template
import tensorflow.lite as tflite
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the TFLite model
MODEL_PATH = 'skin_cancer_model/model.tflite'
model_path = os.path.join(os.getcwd(), MODEL_PATH)
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(filepath)
            print(f"File saved at: {filepath}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({'error': 'Failed to save file'}), 500
        
        # Preprocess the image and make a prediction
        image = Image.open(filepath).resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0).astype(np.float32)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]  # Get the first prediction
        
        # Define the labels and names
        labels = [
            {"label": "akiec", "name": "Actinic Keratoses and Intraepithelial Carcinomae"},
            {"label": "bcc", "name": "Basal Cell Carcinoma"},
            {"label": "bkl", "name": "Benign Keratosis"},
            {"label": "df", "name": "Dermatofibroma"},
            {"label": "mel", "name": "Melanoma"},
            {"label": "nv", "name": "Melanocytic Nevi"},
            {"label": "vasc", "name": "Vascular Skin Lesions"}
        ]
        
        # Find the predicted class
        predicted_index = np.argmax(prediction)  # Get the index of the highest probability
        predicted_label = labels[predicted_index]  # Get the corresponding label and name

        # Return the predicted_name in the response
        return jsonify({
            'message': 'File uploaded successfully',
            'filepath': filepath,
            'prediction': prediction.tolist(),
            'predicted_label': predicted_label['label'],
            'predicted_name': predicted_label['name']
        })

if __name__ == '__main__':
    app.run(debug=True)