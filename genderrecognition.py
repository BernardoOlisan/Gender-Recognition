'''
Real-time Face Gender Recognition using Conv-Nueral Network (CNN) and Cv2

Here we predict the save model that it is train
'''
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# load the model
model = load_model('C:/Users/berna/Desktop/Programming/AI_ML_DL/Projects/FaceGenderRecognition/gender_detection.model')

# open webcams and initiate the camara
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

classes = ['hombre', 'mujer']

# loop through frames
while webcam.isOpened():
    # read frame from webcam
    status, frame = webcam.read()
    webcam.set(cv2.CAP_PROP_FPS, 1000)
    frame = cv2.flip(frame, 1)

    # apply face detection
    face, confidence = cv.detect_face(frame) # this detects that there is a face in the camara, cvlib does, but not if it is a man that detects the neural network

    # loop through detected faces
    for idx, f in enumerate(face):
        # get corner points of face rectangle
        # this only will draw a rectangle when the cvlib detects the face with the vars giving up there
        startX, startY = f[0], f[1]
        endX, endY = f[2], f[3]

        # draw the rectangle over the face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
        # crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection face with the model
        conf = model.predict(face_crop)[0]

        # get label with max acc
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above the face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

    # display output
    cv2.imshow("Gender Detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#realese resources
webcam.release()
cv2.destroyAllWindows()
