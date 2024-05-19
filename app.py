from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import sys

app = Flask(__name__)

# Define a route to the homepage
@app.route('/')
def home():
    return render_template('index.html',result=None)

# Define a route to handle the image classification
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the POST request
    img_file = request.files['file']

    # Read the image file as a byte stream
    img_bytes = img_file.read()

    # Convert the byte stream to a PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    # Preprocess the image
    img = img.resize((150, 150))
    img = np.asarray(img)
    img = img.reshape((1, 150,150,3))
    img = img.astype('float32')
    img = img / 255.0

        # Load the model and predict the class probabilities for the image
    model = load_model('D:\ibm weather\weather_model.h5')

    # Restore the standard output stream to the console
    temp_stdout=io.StringIO()
    sys.stdout = temp_stdout
    prediction = model.predict(img)

    predicted_class_idx= np.argmax(prediction)

    # Map the predicted class index to the actual class name
    class_names = ['.ipynb_checkpoints','alien_test','cloudy','foggy','rainy','shine','sunrise','test.csv']
    predicted_class_name = class_names[predicted_class_idx]
                                       

    # Return the predicted class probabilities as a JSON object
    response = {
        'class_probs': prediction.tolist(),
        'class_name':predicted_class_name
    }

    sys.stdout=sys.__stdout__
    
    return render_template('ouput.html', result=predicted_class_name)

if __name__ == '__main__':
    app.run(debug=True)
