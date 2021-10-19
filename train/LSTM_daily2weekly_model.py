from typing import Dict, Optional

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from main.utils.utils import time_series_data_preparation
from main.model.LSTM_daily2weekly_architecture import build_lstm_v1_architecture, build_lstm_v2_architecture, \
    build_lstm_v3_architecture, build_cnn_lstm_v2_architecture


def build_lstm_v1_model(train: 'np.ndarray', train_label: 'np.ndarray', n_input: int, lstm_units: int,
                        decoder_dense_units: int, optimizer: Optimizer, epochs: int, n_out: int, n_gap: int,
                        day_increment: int, batch_size: int, verbose: int, statistical_operation: Dict) -> Model:
    """
    Args:
        train: model training input features
        train_label: model training input target
        epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
                Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is
                not trained for a number of iterations given by epochs, but merely until the epoch of index epochs
                is reached.
        lstm_units: Positive integer, dimensionality of the output space.
        decoder_dense_units: Positive integer, dimensionality of the output space.
        batch_size: Number of samples per batch.
        verbose: Verbosity mode, 0 or 1.
        n_input: time steps of encoder in days.
        n_out: time steps of decoder in weeks.
        n_gap: time difference between end of encoder and begin of decoder in weeks.
        statistical_operation: specify the method of data aggregation from daily data to weekly data.
        optimizer: Keras optimizers.
        day_increment: day increment of generate new data
    Returns:
        A tensorflow trained model
    """

    # prepare data
    train_x, train_x_weekly, _, train_y_weekly = time_series_data_preparation(train, train_label, n_input,
                                                                              n_out=n_out, n_gap=n_gap,
                                                                              day_increment=day_increment,
                                                                              statistical_operation=statistical_operation)

    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y_weekly.shape[1]

    # define model
    model = build_lstm_v1_architecture(lstm_units, decoder_dense_units, n_timesteps, n_features, n_outputs)

    model.compile(loss='mse', optimizer=optimizer)
    # fit network
    model.fit({'daily_inputs': train_x, 'weekly_inputs': train_x_weekly},
              {'outputs': train_y_weekly}, epochs=epochs, batch_size=batch_size, verbose=verbose,
              shuffle=True)

    return model


def build_lstm_v2_model(train: 'np.ndarray', train_label: 'np.ndarray', n_input: int, lstm_units: int,
                        decoder_dense_units: int, optimizer: Optimizer, epochs: int, n_out: int, n_gap: int,
                        day_increment: int, batch_size: int, verbose: int, statistical_operation: Dict) -> Model:
    """
    Args:
        train: model training input features
        train_label: model training input target
        epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
                Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is
                not trained for a number of iterations given by epochs, but merely until the epoch of index epochs
                is reached.
        lstm_units: Positive integer, dimensionality of the output space.
        decoder_dense_units: Positive integer, dimensionality of the output space.
        batch_size: Number of samples per batch.
        verbose: Verbosity mode, 0 or 1.
        n_input: time steps of encoder in days.
        n_out: time steps of decoder in weeks.
        n_gap: time difference between end of encoder and begin of decoder in weeks.
        statistical_operation: specify the method of data aggregation from daily data to weekly data.
        optimizer: Keras optimizers.
        day_increment: day increment of generate new data
    Returns:
        A tensorflow trained model
    """

    # prepare data
    train_x, _, _, train_y_weekly = time_series_data_preparation(train, train_label, n_input,
                                                                 n_out=n_out, n_gap=n_gap,
                                                                 day_increment=day_increment,
                                                                 statistical_operation=statistical_operation)

    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y_weekly.shape[1]

    # define model
    model = build_lstm_v2_architecture(lstm_units, decoder_dense_units, n_timesteps, n_features, n_outputs)

    model.compile(loss='mse', optimizer=optimizer)
    # fit network
    model.fit({'daily_inputs': train_x},
              {'outputs': train_y_weekly}, epochs=epochs, batch_size=batch_size, verbose=verbose,
              shuffle=True)

    return model


def build_lstm_v3_model(train: 'np.ndarray', train_label: 'np.ndarray', n_input: int, lstm_units: int,
                        decoder_dense_units: int, optimizer: Optimizer, epochs: int, n_out: int, n_gap: int,
                        day_increment: int, batch_size: int, verbose: int, statistical_operation: Dict) -> Model:
    """
    Args:
        train: model training input features
        train_label: model training input target
        epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
                Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is
                not trained for a number of iterations given by epochs, but merely until the epoch of index epochs
                is reached.
        lstm_units: Positive integer, dimensionality of the output space.
        decoder_dense_units: Positive integer, dimensionality of the output space.
        batch_size: Number of samples per batch.
        verbose: Verbosity mode, 0 or 1.
        n_input: time steps of encoder in days.
        n_out: time steps of decoder in weeks.
        n_gap: time difference between end of encoder and begin of decoder in weeks.
        statistical_operation: specify the method of data aggregation from daily data to weekly data.
        optimizer: Keras optimizers.
        day_increment: day increment of generate new data
    Returns:
        A tensorflow trained model
    """

    # prepare data
    train_x, train_x_weekly, _, train_y_weekly = time_series_data_preparation(train, train_label, n_input,
                                                                              n_out=n_out, n_gap=n_gap,
                                                                              day_increment=day_increment,
                                                                              statistical_operation=statistical_operation)

    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y_weekly.shape[1]

    # define model
    model = build_lstm_v3_architecture(lstm_units, decoder_dense_units, n_timesteps, n_features, n_outputs)

    model.compile(loss='mse', optimizer=optimizer)
    # fit network
    model.fit({'daily_inputs': train_x, 'weekly_inputs': train_x_weekly},
              {'outputs': train_y_weekly}, epochs=epochs, batch_size=batch_size, verbose=verbose,
              shuffle=True)

    return model


def build_cnn_lstm_v2_model(train: 'np.ndarray', train_label: 'np.ndarray', n_input: int, lstm_units: int,
                            decoder_dense_units: int, conv1d_filters, optimizer: Optimizer, epochs: int, n_out: int,
                            n_gap: int, day_increment: int, batch_size: int, verbose: int,
                            statistical_operation: Dict) -> Model:
    """
    Args:
        train: model training input features
        train_label: model training input target
        epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
                Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is
                not trained for a number of iterations given by epochs, but merely until the epoch of index epochs
                is reached.
        lstm_units: Positive integer, dimensionality of the output space.
        decoder_dense_units: Positive integer, dimensionality of the output space.
        conv1d_filters: Integer, the dimensionality of the output space.
        batch_size: Number of samples per batch.
        verbose: Verbosity mode, 0 or 1.
        n_input: time steps of encoder in days.
        n_out: time steps of decoder in weeks.
        n_gap: time difference between end of encoder and begin of decoder in weeks.
        statistical_operation: specify the method of data aggregation from daily data to weekly data.
        optimizer: Keras optimizers.
        day_increment: day increment of generate new data
    Returns:
        A tensorflow trained model
    """

    # prepare data
    train_x, _, _, train_y_weekly = time_series_data_preparation(train, train_label, n_input, n_out=n_out,
                                                                 n_gap=n_gap, day_increment=day_increment,
                                                                 statistical_operation=statistical_operation)

    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y_weekly.shape[1]

    # define model
    model = build_cnn_lstm_v2_architecture(lstm_units, decoder_dense_units, conv1d_filters,
                                           n_timesteps, n_features, n_outputs)

    model.compile(loss='mse', optimizer=optimizer)
    # fit network
    model.fit({'daily_inputs': train_x},
              {'outputs': train_y_weekly}, epochs=epochs, batch_size=batch_size, verbose=verbose,
              shuffle=True)

    return model