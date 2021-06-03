import PIL
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/")
def hello():
    return "Use /predict to get predictions"


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    # print(type(data))
    # image = PIL.Image.open(data)
    # image_arr = keras.preprocessing.image.img_to_array(image)
    # print(type(input_arr), input_arr.shape)
    predict = model_prediction(file)
    return jsonify({"result": predict})


def model_prediction(file):
    IMG_SIZE = 300

    image = PIL.Image.open(file)
    image_arr = keras.preprocessing.image.img_to_array(image)
    image_arr = tf.image.resize(image_arr, [IMG_SIZE, IMG_SIZE])
    image_arr = np.array([image_arr])

    predictions = model.predict(image_arr, verbose=1)
    predictions = np.argmax(predictions, axis=-1)  # multiple categories
    return int(predictions[0])


def load_model():
    PATH = 'model.hdf5'
    return tf.keras.models.load_model(PATH)


model = load_model()
print(__name__)

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)
