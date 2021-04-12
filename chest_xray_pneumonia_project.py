import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pydicom

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D,MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD


# def get_data():
#     data_dir = pathlib.Path('cleaned_train_images')
#
# 	train_rescale = ImageDataGenerator(rescale=1./255,validation_split=0.2,zoom_range=0.15)
# 	val_rescale = ImageDataGenerator(rescale=1./255)
#
# 	train_generator = train_rescale.flow_from_directory(directory=data_dir,subset= 'training',batch_size=32,class_mode= 'binary')


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
					 kernel_initializer='he_uniform',
					 input_shape=(256, 256, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print("define_model finished")
    return model

def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)

	return scores, histories

def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()

def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
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
														color_mode= 'grayscale')
	val_generator = val_rescale.flow_from_directory(directory=data_dir,
													subset='validation',
													batch_size=32,
													class_mode='binary',
													color_mode= 'grayscale')
	print(val_generator.class_indices)
	print(val_generator.image_shape)

# print('got_data finished')
    # model = define_model()
    # print('define model finished')
    # scores, histories = evaluate_model(X_train, y_train, n_folds=5)
    # print('evaluate model finished')
    # summarize_diagnostics(histories)
    # summarize_performance(scores)

if __name__=='__main__':
    main()
