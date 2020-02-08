import numpy as np
import pandas as pd
import cv2

'''
read csv file
generate list of pixels
declare faces list
reshape face as array
resize into 48x48 dimension
append into faces list
convert faces list into faces array
return expanded dimension of faces array
return emotions matrix carrying emotion column data of fer2013
'''

def load_dataset():
    data = pd.read_csv('data/fer2013.csv')
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_array in pixels:
        face = [int(pixel) for pixel in pixel_array.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (48,48))
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions

'''
input is x
x converted to float
'''
def input_processing(x, v2=True):
    x = x.astype('float32')
    x = x/255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x 