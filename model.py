from keras.datasets import cifar10

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data() 
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255
    x_test = x_test / 255

    y_train = np.utils.to_catagorical(y_train)
    y_test = np.utils.to_catagorical(y_test)

    return x_train, y_train, x_test, y_test


def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same',
              activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax')

    return model


def train_model(model):
    
    x_train, y_train, x_test, y_test = load_data()
    model = get_model()

    sgd = SGD(lr=0.01, momentum=0.9, decay=(0.01/25), nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])

    return model


def get_trained_model():

    model = load_model('cfar_model_20_epochs.h5')

    return model
    
