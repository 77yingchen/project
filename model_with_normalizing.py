from sklearn.preprocessing import normalize
import numpy as np
from keras import models
from keras import layers
import random

random.seed(0)
np.random.seed(0)

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

mean = np.mean(y_train)
std = np.std(y_train)

y_train = (y_train-mean)/std

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

y_test = (y_test-mean)/std

# y_train = normalize(y_train[:,np.newaxis], axis=0).ravel()

# y_test = normalize(y_test[:,np.newaxis], axis=0).ravel()

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1))
model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

model.fit(x_train, y_train, epochs=150, batch_size=32, verbose=2, shuffle = True)

val_mse, val_mae = model.evaluate(x_test, y_test, verbose=0)

print val_mae