import cv2
import os
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical


image_path = "Dataset/"
dataset = []
label = []

inp_size = 64

no_tumor_images = os.listdir(image_path + 'no/')
yes_tumor_images = os.listdir(image_path + 'yes/')

for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_path+'no/'+image_name)  # matrix
        image = Image.fromarray(image, 'RGB')  # pil image
        image = image.resize((inp_size, inp_size))
        dataset.append(np.array(image))  # append to dataset
        label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_path+'yes/'+image_name)  # matrix
        image = Image.fromarray(image, 'RGB')  # pil image
        image = image.resize((inp_size, inp_size))
        dataset.append(np.array(image))  # append to dataset
        label.append(1)


dataset = np.array(dataset)
label = np.array(label)

X_train, X_test, y_train, y_test = train_test_split(
    dataset, label, test_size=0.2, random_state=0)

x_train = normalize(X_train, axis=1)
x_test = normalize(X_test, axis=1)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)


model = Sequential()


model.add(Conv2D(32, (3, 3), input_shape=(inp_size, inp_size, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))


# Binary Crossentropy  = 1 ,sigmoid
#categoryCrossentropy = 2 ,softmax

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1,
          validation_data=(x_test, y_test), shuffle=False)

model.save('Brain_Tumor_Detection10epochCategorical.h5')

print("Completeee")
