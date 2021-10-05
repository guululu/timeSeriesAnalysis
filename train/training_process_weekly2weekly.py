from copy import copy
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from train.LSTM_weekly2weekly_model import build_lstm_v2_model, build_cnn_lstm_v2_model, \
    build_cnn_lstm_v2_luong_model, build_lstm_feed_model, build_conv2d_lstm_v2_model


def forecast(model: Model, history: List, n_input: int,
             statistical_operation: Optional[Dict] = None):

    # flatten data
    data = np.array(history)
    data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

    input_weekly = []

    for col_index, operations in statistical_operation.items():
        for operation in operations:
            statistical_feature = eval(f'np.nan{operation}(np.array(np.split(data[-n_input:, '
                                       f'col_index], n_input // 7)), axis=1, keepdims=True)')
            if np.isnan(np.sum(statistical_feature)):
                statistical_feature = np.nan_to_num(statistical_feature)
            input_weekly.append(statistical_feature)

    input_weekly = np.expand_dims(np.concatenate(input_weekly, axis=1), 0)

    if len(model.input.shape) == 5:
        input_weekly = input_weekly.reshape(input_weekly.shape[0], 2, 1, 4, input_weekly.shape[-1])
    # forecast the next week
    yhat = model.predict(input_weekly, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


def evaluation_model(model: Model, train, test, label_test,  n_input: int, n_out: int = 4,
                     n_gap: int = 7, statistical_operation: Optional[Dict] = None):

    history = [x for x in train[:-n_gap]]  # [x for x in train[:-(n_out + n_gap)]]

    predictions = list()
    observations = list()

    for i in range(len(test) - n_gap):
        if (i + n_out + n_gap) * 7 <= len(label_test):
            yhat_sequence = forecast(model, history, n_input, statistical_operation=statistical_operation)
            predictions.append(yhat_sequence)
            observation = np.split(label_test[(i + n_gap) * 7: (i + n_out + n_gap) * 7], n_out)
            observations.append(np.array(observation).sum(axis=1))
            history.append(test[i, :])

    predictions = np.array(predictions)[:, :, 0]
    observations = np.array(observations)

    return predictions, observations


def training_process(daily_data: pd.DataFrame, epochs, decoder_dense_units, learning_rate, beta_1, beta_2, epsilon,
                     model_name: str, statistical_operation: Optional[Dict] = None, scaler=None,
                     lstm_units: Optional[int] = None, conv1d_filters: Optional[int] = None,
                     n_input: Optional[int] = None, n_out: Optional[int] = None, n_gap: Optional[int] = None,
                     scale: Optional[int] = None):

    tscv = TimeSeriesSplit(n_splits=13, test_size=n_out * 7, max_train_size=52 * 7)

    predictions = []
    observations = []

    if conv1d_filters:
        conv1d_filters = int(conv1d_filters)

    if lstm_units:
        lstm_units = int(lstm_units)

    decoder_dense_units = int(decoder_dense_units)
    epochs = int(epochs)

    for train_index, val_index in tscv.split(daily_data):

        train = daily_data.iloc[train_index].values / scale
        val = daily_data.iloc[np.concatenate((train_index[-(n_gap * 7):], val_index))].values / scale

        val_label = val[:, 0]
        train_label = train[:, 0]

        if scaler:
            train = scaler.fit_transform(train)
            val = scaler.transform(val)

        val = np.array(np.split(val, len(val) / 7))
        train = np.array(np.split(train, len(train) / 7))

        optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

        if model_name == 'lstm_v2':
            model = build_lstm_v2_model(train, train_label, n_input=n_input, lstm_units=lstm_units,
                                        decoder_dense_units=decoder_dense_units, optimizer=optimizer,
                                        epochs=epochs, n_out=n_out, n_gap=n_gap,
                                        statistical_operation=statistical_operation)
        if model_name == 'cnn_lstm_v2':
            model = build_cnn_lstm_v2_model(train, train_label, n_input=n_input, lstm_units=lstm_units,
                                            decoder_dense_units=decoder_dense_units, conv1d_filters=conv1d_filters,
                                            optimizer=optimizer, epochs=epochs, n_out=n_out, n_gap=n_gap,
                                            statistical_operation=statistical_operation)
        if model_name == 'cnn_lstm_v2_luong':
            model = build_cnn_lstm_v2_luong_model(train, train_label, n_input=n_input, lstm_units=lstm_units,
                                                  decoder_dense_units=decoder_dense_units,
                                                  conv1d_filters=conv1d_filters, optimizer=optimizer,
                                                  epochs=epochs, n_out=n_out, n_gap=n_gap,
                                                  statistical_operation=statistical_operation)
        if model_name == 'lstm_feed':
            model = build_lstm_feed_model(train, train_label, n_input=n_input, lstm_units=lstm_units,
                                          decoder_dense_units=decoder_dense_units, optimizer=optimizer,
                                          epochs=epochs, n_out=n_out, n_gap=n_gap,
                                          statistical_operation=statistical_operation)

        if model_name == 'conv2d_lstm_v2':
            model = build_conv2d_lstm_v2_model(train, train_label, n_input=n_input, filters=conv1d_filters,
                                               decoder_dense_units=decoder_dense_units,  optimizer=optimizer,
                                               epochs=epochs, n_out=n_out, n_gap=n_gap,
                                               statistical_operation=statistical_operation)

        p, o = evaluation_model(model, train, val, val_label, n_input, n_out=n_out, n_gap=n_gap,
                                statistical_operation=statistical_operation)

        observations.append(o)
        predictions.append(p)

    observations = np.concatenate(observations, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    mse = mean_squared_error(observations, predictions)

    return -mse