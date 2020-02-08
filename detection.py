import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils

face_detection = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('meta-data/Weight.hdf5', compile=False)

EMOTIONS = ["ANGRY", "DISGUST", "SCARED", "HAPPY", "SAD", "SURPRISED", "NEUTRAL"]

cv2.namedWindow("FACE DETECTION WINDOW")
video = cv2.VideoCapture(0)
while True:
    frame = video.read()[1]
    frame = imutils.resize(frame, width = 300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, 
    minNeighbors=5, 
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
    )

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    cloned_frame = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))[0]
        (face_x, face_y, face_width, face_height) = faces

        roi = gray[face_y:face_y + face_height, face_x:face_x + face_width]
        roi = cv2.resize(roi, (64,64))
        roi = roi.astype("float")/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = emotion_classifier.predict(roi)[0]
        probability = np.max(prediction)
        label = EMOTIONS[prediction.argmax()]
    else:
        continue
    
    for(i, (emotion, probability)) in enumerate(zip(EMOTIONS, prediction)):
        
        text_label = "{}: {:.2f}%".format(emotion, probability * 100)

        width = int(probability * 100)
        cv2.rectangle(canvas, (7, (i * 35) + 5), (width, (i*35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text_label, (10, (i * 35)+23), cv2.FONT_HERSHEY_PLAIN, 0.45, (255,255,255), 2)
        cv2.putText(cloned_frame, label, (face_x, face_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
        cv2.rectangle(cloned_frame, (face_x, face_y), (face_x + face_width, face_y+face_height), (0,0,255), 2)
    
    cv2.imshow('SHOWING FACE', cloned_frame)
    cv2.imshow("PROBABILITIES", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    video.release()
    cv2.destroyAllWindows()