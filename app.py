
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

from keras.models import load_model
model = load_model("skin_cancer_model.h5")
print("Model loaded successfully.")

class_names = ['Benign', 'Malignant']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register_page.html')
def register_page():
    return render_template('register_page.html')

@app.route('/success.html')
def success():
    return render_template('success.html')

@app.route('/check.html')
def check():
    return render_template('check.html') 


@app.route('/check', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(180, 180))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    result = class_names[class_index]
    confidence = float(prediction[class_index])


    return render_template('check.html', result=result, confidence=confidence*100)

@app.route('/skin_cancer_types.html')
def skin_cancer_info():
    return render_template('skin_cancer_types.html')

if __name__ == '__main__':
    app.run(debug=True)







