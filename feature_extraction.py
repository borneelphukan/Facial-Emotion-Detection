import numpy as np
import pandas as pd
import cv2

data_path = 'Data/fer2013.csv'
img_size = (48, 48)

def load_data():
    data = pd.read_csv(data_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_seq in pixels:
        face = [int(pixel) for pixel in pixel_seq.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), img_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    face_emotions = pd.get_dummies(data['emotion']).values
    return faces, face_emotions

def preprocessing(x, v2 = True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x