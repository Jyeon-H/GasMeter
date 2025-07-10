import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def train_mlp(x, y, epochs=30, batch_size=128):

    x = x.reshape(-1, 28*28).astype(np.float32) / 255.0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)


    n_input = 784
    n_hidden = 1024
    n_output = 10

    model = Sequential([
    Dense(1024, activation='tanh', input_shape=(784,), 
    kernel_initializer='random_uniform', bias_initializer='zeros'),
    Dense(10, activation='tanh', 
    kernel_initializer='random_uniform', bias_initializer='zeros')
    ])
    
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
    validation_data=(x_test, y_test), verbose=2)

    res = model.evaluate(x_test, y_test, verbose=0)
    print("정확도 ", res[1]*100)

    return model, x_test, y_test