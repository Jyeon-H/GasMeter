import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def train_cnn(x, y, epochs=30, batch_size=128):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42) 

    x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    #LeNet-5 유사
    model = Sequential([
        Conv2D(6,(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)),
        Conv2D(16, (5,5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(32, (5,5), padding='same', activation='relu'),
        Conv2D(64, (5,5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(120, (5,5), padding='same', activation='relu'),
        Flatten(),
        Dense(84, activation='relu'),
        Dense(10, activation='softmax')
        ])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    hist = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_test, y_test), verbose=2)

    res = model.evaluate(x_test, y_test, verbose=0)
    print("정확도:", res[1]*100)
    
    return model, x_test, y_test