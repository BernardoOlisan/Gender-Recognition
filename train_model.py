'''
Real-time Face Gender Recognition using Conv-Nueral Network (CNN) and Cv2

Here we train and extract the data, data contains 2000 images per gender
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob

# Initital parameters, and data classification
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96,96,3) # all imgs have to be the same size
# we define 96x96 px but the 3 is the channels, channels is bc we have it in rgb color and not in grayscale 3 primary rgb

data = [] # if data is a man
labels = [] # label man goes here

# loading the image files from the dataset
image_files = [f for f in glob.glob(r'C:\Users\berna\Desktop\Programming\AI_ML_DL\Projects\FaceGenderRecognition\gender_dataset_face'+'/**/*', recursive=True) if not os.path.isdir(f)]
# in image files tuple we loop for all man and woman images, but we need to suffle it
random.shuffle(image_files)

# converting images to arrays and appending to labels the categories
for img in image_files:
    image = cv2.imread(img)

    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)

    data.append(image)

    label = img.split(os.path.sep)[-2]
    if label == "woman":
        label = 1
    else:
        label = 0

    labels.append([label])

# pre-procrssing,  convert all data into arrays
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)

trainY = to_categorical(trainY, num_classes=2) # [[1,0], [0,1], [0,1] .....]
testY = to_categorical(testY, num_classes=2)

# Argumenting the dataset
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")


# Define the model
def build(width, height, depth, classes): # width and height is 96x96 px we define. the depth is the channels = 3, and classes the categoriees 1 or 0 woman or man
    model = Sequential()
    inputShape = height, width, depth
    chanDim = -1

    if K.image_data_format() == "channels_first": # Retunrs a string if channels first or channels last
        inputShape = depth, height, width
        chanDim = 1


    # Creating the Neural Network(CNN) 2D
    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes)) # The prediction 0 or 1 woman or man two neurons at the end
    model.add(Activation("sigmoid"))

    return model

# Build the model and call it assinging it with the parameters on the funct
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)

# compile the model and fitting it
opt = Adam(lr=lr, decay=lr/epochs)
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])

# fitting (training funct) the model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1)

# save the train model
model.save('gender_detection.model')


# NOW WE USE MATPLOTLIB TO PLOT THE ACCURACY LOSS AND VALIDATION TO SEE IF IT IS GOOD INSTEAD OF USING TENSORBOARD
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# Save the plot img to see result
plt.savefig('Results.png')
