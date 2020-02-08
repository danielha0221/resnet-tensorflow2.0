from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

class Datasets():
  
  def __init__(self):

  def preproc(self, x_train, y_train, x_test, y_test):
    x_train = x_train / 255
    x_test = x_test / 255
    y_cat_train = to_categorical(y_train, 10)
    y_cat_test = to_categorical(y_test, 10)

    return x_train, y_cat_train, x_test, y_cat_test

  def get_data(self):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_cat_train, x_test, y_cat_test = self.preproc(x_train, y_train, x_test, y_test)

    return  x_train, y_cat_train, x_test, y_cat_test