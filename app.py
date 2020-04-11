from flask import Flask, render_template, request

#from __future__ import division#, print_function

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions



# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#import covid19 model
from COVID19_VIRUS_DETECTOR import covid19_ai_diagnoser

CONSTANT_DIAGNOSIS_IMAGE_SPAN = 480


# Define a flask app
app = Flask(__name__)

#Ruta 
@app.route('/')
def index():
    return render_template('index.html')

#---------------------------------------------------------------------
#---------------------------------------------------------------------
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------


def loadCovid19ImageFromName(filename):
	DIAGNOSIS_RESULT = covid19_ai_diagnoser.doOnlineInference_covid19Pneumonia("/uploads/" + filename + "/")
	
	#DIAGNOSIS_RESULT += "**Non-Covid19 Mode Result**\n" + filename+"\n\n"
	#DIAGNOSIS_RESULT += covid19_ai_diagnoser.doOnlineInference_covid19Pneumonia("\\uploads\\" + filename + "\\")
	print(DIAGNOSIS_RESULT)
	return DIAGNOSIS_RESULT
    


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        #print(f.filename)
        # Save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        currdir = os.getcwd()
        file_path = os.path.join(currdir + '/uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        prediction = loadCovid19ImageFromName(f.filename)

        return prediction
    return None



# main
if __name__ == "__main__":
	app.run(debug=True, port=5000)