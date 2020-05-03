import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

# set to your directory
datadir = "C:\\Users\\nickm\\PycharmProjects\\CatDogPredictor\\PetImages"
categories = ['Dog', 'Cat']
img_size = 50
training_data = []
load_data = Trueq
x = [] # features
y = [] # labels

def load_training_data():
    global training_data, x, y
    with open("data", "rb") as file:
        training_data = pickle.load(file)
        random.shuffle(training_data)
    with open("x_pickle", "rb") as file:
        x = pickle.load(file)
    with open("y_pickle", "rb") as file:
        y = pickle.load(file)

def create_training_data():
    global training_data, x, y
    for caetegoir in categories: # rip spelling lol
        path = os.path.join(datadir, caetegoir)
        class_num = categories.index(caetegoir)
        for img in os.listdir(path):
            try:
                img_array =  cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                os.remove(path+"\\"+img)

    for features, label in training_data:
        x.append(features)
        y.append(label)
    x = np.array(x).reshape(-1, img_size, img_size, 1)

    # dump
    with open("x_pickle", "wb") as file:
        pickle.dump(x, file)
    with open("y_pickle", "wb") as file:
        pickle.dump(y, file)
    with open("data", "wb") as file:
        pickle.dump(training_data, file)
    random.shuffle(training_data)


# if you havent saved a model or training data create it all here
if not load_data:
    create_training_data()

    # normalize
    x = np.array(x / 255.0)
    y = np.array(y)

    # create our model
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(x, y, batch_size=32, epochs=3, validation_split=0.1)

    # save model
    model.save('saved_model/')
else:
    # load data and model
    load_training_data()
    model = tf.keras.models.load_model('saved_model/saved_model.pb')
