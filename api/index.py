from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

model_with_trained_path = os.path.join(os.path.dirname(__file__), 'models', 'skin_with_model.h5')
model_with_trained_model = tf.keras.models.load_model(model_with_trained_path)
classes = ['Actinic keratosis', 'Basal cell carcinoma', 'Benign keratosis', 'Dermatofibroma', 'Melanocytic nevus', 'Melanoma', 'Tinea Ringworm Candidiasis']

# Symptom weights for each class
weights = {
    "Actinic keratosis": [0.8, 0.7, 0.3, 0.2],
    "Basal cell carcinoma": [0.6, 0.75, 0.5, 0.1],
    "Benign keratosis": [0.4, 0.2, 0.2, 0.8],
    "Melanoma": [0.9, 0.85, 0.7, 0.6],
    "Melanocytic nevus": [0.2, 0.4, 0.1, 0.8],
    "Dermatofibroma": [0.5, 0.6, 0.2, 0.7],
    "Tinea Ringworm Candidiasis": [0.1, 0.8, 0.9, 0.5]
}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    symptoms = {
        'rough_texture': int(request.form.get('rough_texture', 0)),
        'redness': int(request.form.get('redness', 0)),
        'itchiness': int(request.form.get('itchiness', 0)),
        'lesions': int(request.form.get('lesions', 0))
    }
    
    model = model_with_trained_model

    # Process image
    image = Image.open(file)
    image = image.resize((224, 224))
    image_rgb = Image.new("RGB", image.size)
    image_rgb.paste(image)
    image_array = np.array(image_rgb) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Get model predictions
    predictions = model.predict(image_array)[0]
    
    # Calculate combined probabilities
    combined_probabilities = []
    for i, predicted_class in enumerate(classes):
        model_confidence = predictions[i]
        symptom_weights = weights.get(predicted_class)
        print(predicted_class)
        # Check if symptom_weights is None and handle it
        if symptom_weights is None:
            print(f"Warning: No symptom weights found for class '{predicted_class}'. Using default weights.")
            symptom_weights = [0, 0, 0, 0]  # Default weights if none are found

        symptom_probability = (
            symptom_weights[0] * symptoms['rough_texture'] +
            symptom_weights[1] * symptoms['redness'] +
            symptom_weights[2] * symptoms['itchiness'] +
            symptom_weights[3] * symptoms['lesions']
        )
        combined_confidence = (symptom_probability * 0.5) + (model_confidence * 0.5)
        combined_probabilities.append(combined_confidence)

    final_probabilities = softmax(np.array(combined_probabilities))
    result = {predicted_class: final_probabilities[i] for i, predicted_class in enumerate(classes)}

    # Find the highest probability class
    max_class = max(result, key=result.get)
    max_confidence = result[max_class]

    return jsonify({
        'predicted_class': max_class,
        'max_confidence': max_confidence,
        'class_probabilities': result
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4000)
