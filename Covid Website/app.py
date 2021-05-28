#Import required libraries 
from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
import tensorflow as tf
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Loading h5 file containing  saved model based on experiments from juptyer notebook
CovidModel = tf.keras.models.load_model('static/CovidModelSave.h5')

COUNT = 0

#Initialise the flask application
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

#render index.html as the first page 
@app.route('/')
def man():
    return render_template('index.html')

#Prediction html page will be rendered upon image upload 
@app.route('/prediction', methods=['POST'])
def home():
    global COUNT

    img = request.files['image']      #accesssing user uploaded file image           
    img.save('static/{}.png'.format(COUNT))      #save image within static folder          
    img_array = cv2.imread('static/{}.png'.format(COUNT))     #storing image in array          

    img_array = cv2.resize(img_array, (60,60))     #resize image to (60,60) to match our model input shape           
    img_array = img_array / 255.0      #normalise pixel values            
    img_array = img_array.reshape(1,60,60,3)               

    prediction = CovidModel.predict(img_array)      #Predict image class using our model         


    #Store predicitons of both options (covid or no covid) in x and y
    x = round(prediction[0,0], 2)  
    y = round(prediction[0,1], 2)
    results_array = np.array([x,y])
    COUNT += 1

    #Load prediction.html with our results 
    return render_template('prediction.html', data=results_array)               


#Loads the image the user uploaded 
@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.png".format(COUNT-1))

#Add an about.html page
@app.route('/about.html')
def about():
    return render_template('about.html')

#Add an info.html page
@app.route('/info.html')
def info():
    return render_template('info.html')

if __name__ == '__main__':
    app.run()
