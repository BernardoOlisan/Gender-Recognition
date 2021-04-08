# The same genderrecognition.py code but with multi-threading to make it faster and fix the the lag of the other one
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# open webcam and initiate the cam
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# opencv class
class VideoStream:
    def __init__(self):
        # read frame from webcam
        self.status, self.frame = webcam.read()
        webcam.set(cv2.CAP_PROP_FPS, 1000)
        self.frame = cv2.flip(self.frame, 1)

        print("videostream working")


# face detection class
class face_detection:
    def __init__(self):
        # use VideoStream Class variables
        self.videostream = VideoStream()
        self.frame = self.videostream.frame

        # apply face detection
        self.face, self.confidence = cv.detect_face(self.frame)

        # loop through detected faces
        for self.idx, self.f in enumerate(self.face):
            # get the corner point of the rectangle
            self.startX, self.startY = self.f[0], self.f[1]
            self.endX, self.endY = self.f[2], self.f[3]

            cv2.rectangle(self.frame, (self.startX, self.startY), (self.endX, self.endY), (0,255,0), 2)
            self.face_crop = np.copy(self.frame[self.startY:self.endY, self.startX:self.endX])

            if self.face_crop.shape[0] < 10 or self.face_crop.shape[1] < 10:
                continue

            # preprocessing for gender detection model
            self.face_crop = cv2.resize(self.face_crop, (96,96))
            self.face_crop = self.face_crop.astype("float") / 255.0
            self.face_crop = img_to_array(self.face_crop)
            self.face_crop = np.expand_dims(self.face_crop, axis=0)

            GFR()

        print("face_detection working")

# gender recognition class
class GFR:
    def __init__(self):
        self.model = load_model("C:/Users/berna/Desktop/Programming/AI_ML_DL/Projects/FaceGenderRecognition/gender_detection.model")
        self.facedetection = face_detection()

        self.face_crop = self.facedetection.face_crop
        self.classes = ['hombre', 'mujer']
        self.startX, self.startY = self.facedetection.startX, self.facedetection.startY
        self.endX, self.endY = self.facedetection.endX, self.facedetection.endY
        self.frame = self.facedetection.frame

        # apply the gender detection face with the model
        self.conf = model.predict(self.face_crop)[0]

        # get label with max acc
        self.idx = np.argmax(self.conf)
        self.label = self.classes[self.idx]

        self.label = "{}: {:.2f}".format(self.label, self.conf[self.idx] * 100)

        self.Y = self.startY - 10 if self.startY - 10 > 10 else self.startY + 10

        # write label and confidence above the face rectangle
        cv2.putText(self.frame, self.label, (self.startX, self.Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

        print("gender recognition working!")


# classes and webcam while loop
gender_detection = GFR()


# loop through frames
while webcam.isOpened():
    VideoStream()
    face_detection()

    # display output
    cv2.imshow("Gender Detection", gender_detection.frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
