#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
#import joblib
import keras.utils as image
from io import BytesIO
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import DenseNet169



#Loading the Model
model = load_model('densenet169.h5', compile=False)

st.image('logo.png')
st.markdown("## Lumpy Skin Disease Identifier with Deep Learning")
st.markdown("""
This app uses Deep learning Model(DENSENET169) to  classify images as either positive or negative for Lumpy Skin Disease.


""")

#Name of Classes

st.markdown("Upload an image of the object")

#Uploading the dog image
object_image = st.file_uploader("Upload an image...", type=['png','jpg','webp'])
submit = st.button('Predict')
#On predict button click
if submit:

    if object_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(object_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR")
        opencv_image = cv2.resize(opencv_image, (224,224))
        opencv_image.shape = (1,224,224,3)
        predictions = model.predict(opencv_image)
        if predictions[0][0]>=0.5:
          st.text('This is an image of:Lumpy Skin ')
            
        elif predictions[0][1]>0.5:
          st.text('This is an image of:normal Skin ')
            
