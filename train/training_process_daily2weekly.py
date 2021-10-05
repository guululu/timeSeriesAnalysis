from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.optimizers import Adam

from train.LSTM_daily2weekly_model import build_lstm_v1_model, build_lstm_v2_model, build_cnn_lstm_v2_model


def forecast(model, history, n_input, n_out):
    # flatten data
    data = np.array(history)
    data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

    input_names = model.input_names

    inputs = []
    if 'daily_inputs' in input_names:
        # retrieve last observations for input data
        input_x = data[-n_input:, :]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        inputs.append(input_x)

    if 'weekly_inputs' in input_names:
        input_x = data[-n_input:, :]
        external = np.array(np.split(input_x, n_out)).sum(axis=1)
        input_weekly = np.expand_dims(external, axis=0)
        inputs.append(input_weekly)

        # forecast the next week
    yhat = model.predict(inputs, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


def evaluation_model(model, train, test, label_test, n_input: int, n_out: int = 4, n_gap: int = 7):

    history = [x for x in train[:-n_gap]]

    predictions = list()
    observations = list()

    for i in range(len(test) - (n_out + n_gap)):
        if (i + n_out + n_gap) * 7 <= len(label_test):
            yhat_sequence = forecast(model, history, n_input, n_out)
            predictions.append(yhat_sequence)
            observation = np.split(label_test[(i + n_gap) * 7: (i + n_out + n_gap) * 7], n_out)
            observations.append(np.array(observation).sum(axis=1))
            history.append(test[i, :])

    predictions = np.array(predictions)[:, :, 0]
    observations = np.array(observations)

    return predictions, observations


def training_process(daily_data: pd.DataFrame, epochs, lstm_units, decoder_dense_units, learning_rate, beta_1, beta_2,
                     epsilon, model_name: str, conv1d_filters: Optional[int] = None, n_input: Optional[int] = None,
                     n_out: Optional[int] = None, n_gap: Optional[int] = None, scale: Optional[int] = None):

    if conv1d_filters:
        conv1d_filters = int(conv1d_filters)

    lstm_units = int(lstm_units)
    decoder_dense_units = int(decoder_dense_units)
    epochs = int(epochs)

    tscv = TimeSeriesSplit(n_splits=13, test_size=n_out * 7, max_train_size=52 * 7)

    predictions = []
    observations = []

    for train_index, val_index in tscv.split(daily_data):

        train = daily_data.iloc[train_index].values / scale
        val = daily_data.iloc[np.concatenate((train_index[-(n_gap * 7):], val_index))].values / scale

        val_label = val[:, 0]
        train_label = train[:, 0]

        optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

        if model_name == 'lstm_v1':
            model = build_lstm_v1_model(train, train_label, n_input=n_input, lstm_units=lstm_units,
                                        decoder_dense_units=decoder_dense_units,
                                        optimizer=optimizer, epochs=epochs, n_out=n_out, n_gap=n_gap)
        if model_name == 'lstm_v2':
            model = build_lstm_v2_model(train, train_label, n_input=n_input, lstm_units=lstm_units,
                                        decoder_dense_units=decoder_dense_units,
                                        optimizer=optimizer, epochs=epochs, n_out=n_out, n_gap=n_gap)
        if model_name == 'cnn_lstm_v2':
            model = build_cnn_lstm_v2_model(train, train_label, n_input=n_input, lstm_units=lstm_units,
                                            decoder_dense_units=decoder_dense_units, conv1d_filters=conv1d_filters,
                                            optimizer=optimizer, epochs=epochs, n_out=n_out, n_gap=n_gap)

        p, o = evaluation_model(model, train, val, val_label, n_input, n_out=n_out, n_gap=n_gap)

        observations.append(o)
        predictions.append(p)

    observations = np.concatenate(observations, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    mse = mean_squared_error(observations, predictions)

    return -mse