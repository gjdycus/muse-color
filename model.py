import csv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

csv_start_idx = 21
csv_end_idx = 26
window_size = 600

def valid_row(row):
    for i in range(csv_start_idx, csv_end_idx):
        if row[i] == "":
            return False
    return True

def scale(x):
    return x / 800

def create_windows(x, y, window_step = 5):
    x_flatten = np.asarray(x).flatten()
    new_x = []
    i = 0
    while (i * window_step) + window_size < len(x_flatten):
        window = []
        for j in range(window_size):
            window.append(x_flatten[i * window_step + j])
        new_x.append(window)
        i += window_step
    new_y = [y] * len(new_x)
    return new_x, new_y

def process_csv(path):
    data = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            if row_count > 0 and valid_row(row):
                fft = np.array([float(i) for i in row[csv_start_idx:csv_end_idx]])
                data.append(scale(fft))
            row_count += 1
    return data

red_x_train = process_csv('./data/v1/red_train.csv')
green_x_train = process_csv('./data/v1/green_train.csv')
red_x_test = process_csv('./data/v1/red_test.csv')
green_x_test = process_csv('./data/v1/green_test.csv')

x_train, y_train = create_windows(red_x_train, 0)
green_x_train, green_y_train = create_windows(green_x_train, 1)
x_test, y_test = create_windows(red_x_test, 0)
green_x_test, green_y_test = create_windows(green_x_test, 1)

for i in green_x_train:
    x_train.append(i)

for i in green_y_train:
    y_train.append(i)

for i in green_x_test:
    x_test.append(i)

for i in green_y_test:
    y_test.append(i)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, input_dim=window_size, activation="relu"),
        tf.keras.layers.Dropout(0.13),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    return model

def train_model():
    model = create_model()

    model.fit(x_train, y_train,
            epochs = 20,
            batch_size = 64)

    [_, score] = model.evaluate(x_test, y_test, batch_size = 64)
    return model, score

def best_model(num):
    best_score = 0
    best_model = None
    for _ in range(num):
        model, score = train_model()
        if score > best_score:
            best_score = score
            best_model = model
    return best_model
