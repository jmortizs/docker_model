# -*-coding:utf-8-*-
import os
import json
import joblib
import numpy as np
import tensorflow as tf

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

class Image(BaseModel):
    data: list

# load model
model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'model.h5'
)
model = tf.keras.models.load_model(model_path)

# load class map
class_map_path = model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    'class_map.pkl'
)
class_map = joblib.load(class_map_path)
class_map = {v:k for k, v in class_map.items()}

# print(type(model.predict(np.random.rand(256, 256).reshape(1, 256, 256, 1))))

# create app
app = FastAPI()

@app.post("/predict")
def predict(data: Image):
    
    # load datga (image)
    img_data = data.data    
    img_data = np.asarray(img_data)
    # predict
    prediction = model.predict(img_data).tolist()
    pred_class = class_map[np.argmax(prediction[0])]
    pred_score = np.amax(prediction[0])
    
    return {'class': pred_class, 'score': float(pred_score)}
