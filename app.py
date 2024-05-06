import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image

base_model = VGG19(include_top=False, input_shape=(128,128,3))
x = base_model.output
flat=Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)
model_03.load_weights('vgg19_model_02.h5')
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "Normal"
	elif classNo==1:
		return "Pulmonary"


def image_prediction(new_image_path):
    test_image = image.load_img(new_image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    # test_image = np.reshape(test_image, (224, 224, 3))
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0
    model_loaded = tensorflow.keras.models.load_model(
    	"my_pulmonary_detection_model.h5")
    prediction = model_loaded.predict(test_image)
    test_image_for_plotting = image.load_img(
    	new_image_path, target_size=(224, 224))
    plt.imshow(test_image_for_plotting)
    if (prediction[0] > 0.5):
        statistic = prediction[0] * 100
        print("This image is %.3f percent %s" %
              (statistic, "P U L M O N A R Y"))
        return 1
    else:
        statistic = (1.0 - prediction[0]) * 100
        print("This image is %.3f percent %s" % (statistic, "N O R M A L"))
        return 0


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((128, 128))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model_03.predict(input_img)
    result01=np.argmax(result,axis=1)
    return result01


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = image_prediction(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)