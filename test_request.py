# -*-coding:utf-8-*-
import json
import numpy as np
import requests

api_path = 'http://127.0.0.1:5000/predict'

np.random.seed(42)
fake_img = np.random.rand(256, 256)
fake_img = fake_img.reshape(1, 256, 256, 1)

data = {'data': fake_img.tolist()}

r = requests.post(api_path, json=data)
print(r.text)

white_img = np.zeros((256, 256), dtype=float)
white_img = white_img.reshape(1, 256, 256, 1)
data = {'data': white_img.tolist()}

r = requests.post(api_path, json=data)
print(r.text)