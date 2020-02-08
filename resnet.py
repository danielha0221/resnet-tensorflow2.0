import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

class ResNet():
  def __init__(self, lay_n):
    self.lay_n = lay_n
    self.fl_n = 64

    if lay_n == 34 :
        self.lay_n_arr = [3, 4, 6, 3]

    elif lay_n == 50 :
        self.lay_n_arr = [3, 4, 6, 3]

    elif lay_n == 101 :
        self.lay_n_arr = [3, 4, 23, 3]

    elif lay_n == 152 :
        self.lay_n_arr = [3, 8, 36, 3]

  def res_block(self, inputs, fl_num, k_size):

    x = layers.BatchNormalization()(inputs, True)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(fl_num, k_size, padding="same")(x)
    x = layers.BatchNormalization()(x, True)
    x = layers.ReLU()(x)

    x = layers.Conv2D(fl_num, k_size, padding="same")(x)
    x = layers.Add()([x, inputs])

    return x

  def bottle_res_block(self, inputs, fl_num, k_size):

    x = layers.BatchNormalization()(inputs, True)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(fl_num, k_size, padding="same")(x)
    x = layers.BatchNormalization()(x, True)
    x = layers.ReLU()(x)

    x = layers.Conv2D(fl_num, k_size, padding="same")(x)
    x = layers.Add()([x, inputs])

    return x
    
  def create_model(self):

    inputs = keras.Input(shape=(32,32,3))
    x = layers.Conv2D(self.fl_n * 1, 3, padding="same")(inputs)

    for i in range(self.lay_n_arr[0]):
      x = self.res_block(x, self.fl_n * 1, 3)

    x = layers.Conv2D(self.fl_n * 2, 3, padding="same")(x)
    
    for i in range(self.lay_n_arr[1]):
      x = self.res_block(x, self.fl_n * 2, 3)

    x = layers.Conv2D(self.fl_n * 4, 3, padding="same")(x)

    for i in range(self.lay_n_arr[2]):
      x = self.res_block(x, self.fl_n * 4, 3)

    x = layers.Conv2D(self.fl_n * 8, 3, padding="same")(x)

    for i in range(self.lay_n_arr[3]):
      x = self.res_block(x, self.fl_n * 8, 3)

    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    return model

  def train(self):
    model = self.create_model()

    model.compile(loss = keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.Accuracy()])
    
    model.fit(x_train, y_cat_train, batch_size=100, epochs=15, validation_data=(x_test, y_cat_test))

    return
    