from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
from keras.applications import DenseNet169
from keras.models import load_model
import numpy as np


app = Flask(__name__)
model = load_model("C:/Users/h p/Desktop/zzzzzzzzzzzzz/Model/emotion_dete.h5")

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = np.ndarray(shape=(1, 48, 48, 3), dtype=np.float32)
    yhat = model.predict(image)
    number_of_class_float = np.argmax(yhat)
    number_of_class_int = int(number_of_class_float)
    cc = class_names[number_of_class_int]



    return render_template('index.html', prediction= cc)


if __name__ == '__main__':
    app.run(port=5000, debug=True)