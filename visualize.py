from keras.models import load_model
#model = load_model('model.h5')

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#CovNet model using NVIDIA architecture
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(160,320,3)))
print(model.output_shape)
model.add(Cropping2D(cropping=((70,25), (1,1)), dim_ordering='tf'))
print(model.output_shape)
model.add(Convolution2D(24,5,5,border_mode="valid", activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5,border_mode="valid", activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5,border_mode="valid", activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3,border_mode="valid", activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))


print(model.summary())