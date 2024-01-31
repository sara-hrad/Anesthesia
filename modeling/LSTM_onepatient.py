import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def split_sequence(u, cp, E, time_horizon):
    x, y = [], []
    for i in range(len(u)):
        end_index = i + time_horizon
        if end_index > (len(u) - 1):
            break
        seq_x = [[u[i], E[i]] for i in range(i, end_index)]
        seq_y = E[end_index]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)


def create_sequence():
    script_dir = os.path.dirname(__file__)
    file_name = 'Eleveld_PDs/lstm_dataset.csv'
    file_path = os.path.join(script_dir, file_name)
    df = pd.read_csv(file_path)
    u = np.array(df['u'])
    cp = np.array(df['cp'])
    E = np.array(df['E'])
    return u, cp, E


def main():
    u, cp, E = create_sequence()
    u_scaled = u / u.max()
    cp_scaled = cp / cp.max()
    E_scaled = (E - E.min()) / (E.max() - E.min())
    time_horizon = 13
    n_features = 2
    x_train, y_train = split_sequence(u_scaled, cp_scaled, E_scaled, time_horizon)
    # x_input = x_train[100, :, :]
    # print(x_train[1100, :, :], y_train[1100])

    # define model
    model = Sequential()
    model.add(LSTM(3, activation='relu', input_shape=(time_horizon, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    # fit model
    model.fit(x_train, y_train, epochs=200, verbose=1)
    # x_input = x_train[100, :, :]
    # x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
    x_input = x_train
    yhat = model.predict(x_input, verbose=0)
    E_predict = yhat*(E.max() - E.min()) + E.min()
    plt.plot(E_predict, color='r')
    plt.plot(E[40:-1], color='g')
    plt.xlabel('time (seconds)')
    plt.ylabel('DoH (WAV_cns)')
    plt.show()
    # print(yhat.shape())


if __name__ == "__main__":
    main()




