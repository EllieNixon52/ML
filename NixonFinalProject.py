import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="3"

x_train = []
y_train = []

x_validate = []
y_validate = []

skip = 3

files = os.listdir("/home/lcn9195/ML/Dataset/191/Images")

indices = range(0, len(files), skip)

for i in indices:
    img = cv2.imread("/home/lcn9195/ML/Dataset/191/Images/" + files[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_train.append(img)

x_train = np.array(x_train)

with open("/home/lcn9195/ML/Dataset/191/Labels.csv", "r") as doc:
    strings = [line.split(",") for line in doc]

strings = strings[0]

for i in indices:
   y_train.append(int(strings[i]))

y_train = np.array(y_train)

files = os.listdir("/home/lcn9195/ML/Dataset/189/Images")

indices = range(0, len(files), skip)

for i in indices:
    img = cv2.imread("/home/lcn9195/ML/Dataset/189/Images/" + files[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_validate.append(img)

x_validate = np.array(x_validate)

with open("/home/lcn9195/ML/Dataset/189/Labels.csv", "r") as doc:
            strings = [line.split(",") for line in doc]

strings = strings[0]

for i in indices:
    y_validate.append(int(strings[i]))

y_validate = np.array(y_validate)

print("x: " + str(np.shape(x_train)) + " y: " + str(np.size(y_train)))
print("xV: " + str(np.shape(x_validate)) + " yV: " + str(np.size(y_validate))) 

def create_rnn_cnn_model(input_shape):
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_initializer=initializer))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializer))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializer))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Reshape((1, -1)))
    
    model.add(layers.GRU(32, activation = 'relu', recurrent_dropout=0.5, kernel_initializer=initializer))

    model.add(layers.Dense(1, activation='sigmoid'))

    return model

input_shape = (180, 320, 1)

model = create_rnn_cnn_model(input_shape)

model.compile(loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_validate, y_validate))

x_train = []
y_train = []

x_validate = []
y_validate = []