import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.client import device_lib

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

# set to your directory
datadir = "C:\\Users\\nickm\\PycharmProjects\\CatDogPredictor\\PetImages"
categories = ['Dog', 'Cat']
img_size = 50
training_data = []
load_data = True
load_model = False
x = [] # features
y = [] # labels
deprecation._PRINT_DEPRECATION_WARNINGS = False


def load_training_data():
    global training_data, x, y
    with open("data", "rb") as file:
        training_data = pickle.load(file)
        random.shuffle(training_data)
    with open("x_pickle", "rb") as file:
        x = pickle.load(file)
    with open("y_pickle", "rb") as file:
        y = pickle.load(file)
    # normalize
    x = np.array(x / 255.0)
    y = np.array(y)


def create_training_data():
    global training_data, x, y
    for caetegoir in categories: # rip spelling lol
        path = os.path.join(datadir, caetegoir)
        class_num = categories.index(caetegoir)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception:
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


# load data 
if load_data:
    load_training_data()


# load model
if load_model:
    model = tf.keras.models.load_model('saves/saved_model/')


# create data
if not load_data:
    create_training_data()

    # normalize
    x = np.array(x / 255.0)
    y = np.array(y)


# create model
if not load_model:      
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                    name = f"conv-{conv_layer}-nodes-{layer_size}-dense-{dense_layer}"
                    tensor_board = TensorBoard(log_dir=f'logs\\{name}')

                    model = Sequential()

                    model.add(Conv2D(layer_size, (3, 3), input_shape=x.shape[1:]))
                    model.add(Activation("relu"))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                    for l in range(conv_layer-1):
                        model.add(Conv2D(layer_size, (3, 3)))
                        model.add(Activation("relu"))
                        model.add(MaxPooling2D(pool_size=(2, 2)))

                    model.add(Flatten())
                    for l in range(dense_layer):
                        model.add(Dense(layer_size))
                        model.add(Activation("relu"))

                    model.add(Dense(1))
                    model.add(Activation("sigmoid"))

                    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
                    model.fit(x, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensor_board])

                    # save model
                    model.save(f'saves/saved-model-{name}/')