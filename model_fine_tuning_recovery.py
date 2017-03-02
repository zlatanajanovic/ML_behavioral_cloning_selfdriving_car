from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sklearn


import os
import csv

samples = []
#with open('./data/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        samples.append(line)
with open('./data_recovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data_middle/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(1.5*center_angle)
                #Data augmentation Flip
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.5)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from keras.models import load_model

#training
model = Sequential()
model = load_model('model.h5')
history_object = model.fit_generator(train_generator, samples_per_epoch= 2*len(train_samples), validation_data=validation_generator, nb_val_samples=len(2*validation_samples), nb_epoch=5)

model.save('model_fine.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()