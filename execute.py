import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import cv2

model_weight = 'borneel_xception.hdf5'

face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotion_classifier = load_model(model_weight, compile=False)
face_emotions = ["ANGRY" ,"DISGUSTED","FEARFUL", "HAPPY", "SAD", "SURPRISED", "NEUTRAL"]
feeling_faces = []

for (index, emotion) in enumerate(face_emotions):
    feeling_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

camera = cv2.VideoCapture(0)
while True:
    frame1 = camera.read()[1]
    frame1 = imutils.resize(frame1,width=700)
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    chart = np.zeros([300, 500, 3], dtype="float32")
    frame2 = frame1.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (face_x, face_y, face_width, face_height) = faces
        
        roi = gray[face_y:face_y + face_height, face_x:face_x + face_width]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        predicts = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(predicts)
        label = face_emotions[predicts.argmax()]
    else:
        continue
        
    for (i, (emotion, prob)) in enumerate(zip(face_emotions, predicts)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)
                emoji_face = feeling_faces[np.argmax(predicts)]
                width = int(prob * 500)
                cv2.rectangle(chart, (7, (i * 35) + 5), (width, (i * 35) + 35), (0, 255, 0), -1)
                cv2.putText(chart, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255, 0, 0), 2)
                cv2.putText(frame2, label, (face_x, face_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                cv2.rectangle(frame2, (face_x, face_y), (face_x + face_width, face_y + face_height), (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame2)
    cv2.imshow("Emotion Probabilities", chart)
    cv2.imshow("Emoji Representation", emoji_face)
    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()