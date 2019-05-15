import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.resnet50 import ResNet50
from keras.datasets import cifar10
import numpy as np

def one_hot(y, n_classes):
    result = np.zeros((y.shape[0], n_classes))
    for i in range(0, y.shape[0]):
        result[i, y[i]] = 1
    return result

np.random.seed(1000)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

'''
#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=32, input_shape=(32,32,3), kernel_size=(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=96, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(1024, input_shape=(4096,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(512))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(10))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(10))
model.add(Activation('softmax'))
'''

model = ResNet50(include_top=False, weights=None, input_shape=(32, 32, 3), classes=10)

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"]) 

model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test)
