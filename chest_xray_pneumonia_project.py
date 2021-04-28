import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pydicom

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


def define_model():
    model = Sequential()
    # Layer 1
    model.add(Conv2D(32, (5, 5), padding="SAME", activation='relu', input_shape=(256, 256, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    # Layer 2
    model.add(Conv2D(64, (5, 5), padding="SAME", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    # Layer 3
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def summary_graphs_loss(data):
    plt.figure(figsize=(10, 7))
    plt.style.use('seaborn')
    plt.plot(data.history['loss'])
    # plt.plot(data.history['val_loss'])
    plt.legend(['Training'])
    plt.title('Training Losses',fontsize=22)
    plt.xlabel('epoch',fontsize=16)
    plt.show()

def summary_graphs_accuracy(data):
    plt.figure(figsize=(10, 7))
    plt.style.use('seaborn')
    plt.plot(data.history['accuracy'])
    # plt.plot(data.history['val_loss'])
    plt.legend(['Training'])
    plt.title('Training Accuracy', fontsize=22)
    plt.xlabel('epoch',fontsize=16)
    plt.show()


def main():
    data_dir = pathlib.Path('cleaned_train_images')

    train_rescale = ImageDataGenerator(rescale=1. / 255,
                                       validation_split=0.2,
                                       zoom_range=0.2)
    val_rescale = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_rescale.flow_from_directory(directory=data_dir,
                                                        subset='training',
                                                        batch_size=32,
                                                        class_mode='binary',
                                                        color_mode='grayscale')
    val_generator = val_rescale.flow_from_directory(directory=data_dir,
                                                    subset='validation',
                                                    batch_size=32,
                                                    class_mode='binary',
                                                    color_mode='grayscale')

    model = define_model()

    train_model = model.fit(train_generator, epochs=5, validation_data= val_generator)
    try:
        summary_graphs_loss(train_model)
        summary_graphs_accuracy(train_model)
        model.save('pneumonia_model.h5')
    except:
        print('Could not print graphs')

if __name__ == '__main__':
    main()
