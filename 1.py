import keras
from keras.datasets import mnist
tf.keras.layers.experimental.preprocessing.RandomRotation(
    factor, fill_mode='reflect', interpolation='bilinear', seed=None, name=None,
    **kwargs
)
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)
