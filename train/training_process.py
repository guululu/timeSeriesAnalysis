import importlib
from typing import Optional, Dict, List, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from main.settings import train_validation_config


def forecast(model: Model, history: List, n_input: int, statistical_operation: Dict) -> 'np.ndarray':
    """
    Args:
        model: a tensorflow trained model
        history: update-to-date data
        n_input: time steps of encoder in days.
        statistical_operation: specify the method of data aggregation from daily data to weekly data.
    Returns:
        model predictions
    """

    # flatten data
    data = np.array(history)
    data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

    input_names = model.input_names

    inputs = {}

    if 'daily_inputs' in input_names:
        # retrieve last observations for input data
        input = data[-n_input:, :]
        input = input.reshape((1, input.shape[0], input.shape[1]))
        inputs['daily_inputs'] = input

    if 'weekly_inputs' in input_names:
        input = []

        for col_index, operations in statistical_operation.items():
            for operation in operations:
                statistical_feature = eval(f'np.nan{operation}(np.array(np.split(data[-n_input:, '
                                           f'col_index], n_input // 7)), axis=1, keepdims=True)')
                if np.isnan(np.sum(statistical_feature)):
                    statistical_feature = np.nan_to_num(statistical_feature)
                input.append(statistical_feature)

        input = np.expand_dims(np.concatenate(input, axis=1), 0)
        inputs['weekly_inputs'] = input

    # # for ConvLSTM2D
    #     if len(model.input.shape) == 5:
    #         input_weekly = input_weekly.reshape(input_weekly.shape[0], 2, 1, 4, input_weekly.shape[-1])

    yhat = model.predict(x=inputs, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


def evaluation_model(model: Model, train: 'np.ndarray', test: 'np.ndarray', label_test: 'np.ndarray',
                     n_input: int, n_out: int, n_gap: int, statistical_operation: Dict):

    """
    Args:
        model:  a tensorflow trained model
        train: model training input features
        test: model testing input features
        label_test: model testing input target
        n_input: time steps of encoder in days.
        n_out: time steps of decoder in weeks.
        n_gap: time difference between end of encoder and begin of decoder in weeks.
        statistical_operation: specify the method of data aggregation from daily data to weekly data.
    Returns:
        ground truth and model predictions
    """

    history = [x for x in train[:-n_gap]]

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


def model_training(model_name: str, model_type: str, train: 'np.ndarray', train_label: 'np.ndarray', lstm_units: int,
                   decoder_dense_units: int, epochs: int, n_input: int, n_out: int, n_gap: int,
                   statistical_operation: Dict, learning_rate: float, beta_1: float, beta_2: float,
                   epsilon: float) -> Model:
    """
    Args:
        model_name: code for neural network architecture.
        model_type: indication of daily2weekly or weekly2weekly model.
        train: model training input features
        train_label: model training input target
        epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
                Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is
                not trained for a number of iterations given by epochs, but merely until the epoch of index epochs
                is reached.
        lstm_units: Positive integer, dimensionality of the output space.
        decoder_dense_units: Positive integer, dimensionality of the output space.
        learning_rate: A Tensor, floating point value, or a schedule that is a
                       tf.keras.optimizers.schedules.LearningRateSchedule, or a callable that takes no arguments and
                       returns the actual value to use, The learning rate.
        beta_1: A float value or a constant float tensor, or a callable that takes no arguments and returns the actual
                value to use. The exponential decay rate for the 1st moment estimates.
        beta_2: A float value or a constant float tensor, or a callable that takes no arguments and returns the actual
                value to use, The exponential decay rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper
                (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper.
        n_input: time steps of encoder in days.
        n_out: time steps of decoder in weeks.
        n_gap: time difference between end of encoder and begin of decoder in weeks.
        statistical_operation: specify the method of data aggregation from daily data to weekly data.
    Returns:
        A tensorflow trained model
    """

    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    assert model_name in ['lstm_v1', 'lstm_v2', 'lstm_v3']
    m = importlib.import_module(f"train.LSTM_{model_type}_model")

    day_increment = train_validation_config['day_increment']
    batch_size = train_validation_config['batch_size']
    verbose = train_validation_config['verbose']

    if model_name == 'lstm_v1':
        model = m.build_lstm_v1_model(train, train_label, n_input=n_input, lstm_units=lstm_units,
                                      decoder_dense_units=decoder_dense_units, optimizer=optimizer, epochs=epochs,
                                      n_out=n_out, n_gap=n_gap, statistical_operation=statistical_operation,
                                      batch_size=batch_size, day_increment=day_increment, verbose=verbose)
    if model_name == 'lstm_v2':
        model = m.build_lstm_v2_model(train, train_label, n_input=n_input, lstm_units=lstm_units,
                                      decoder_dense_units=decoder_dense_units, optimizer=optimizer, epochs=epochs,
                                      n_out=n_out, n_gap=n_gap, statistical_operation=statistical_operation,
                                      batch_size=batch_size, day_increment=day_increment, verbose=verbose)
    if model_name == 'lstm_v3':
        model = m.build_lstm_v3_model(train, train_label, n_input=n_input, lstm_units=lstm_units,
                                      decoder_dense_units=decoder_dense_units, optimizer=optimizer, epochs=epochs,
                                      n_out=n_out, n_gap=n_gap, statistical_operation=statistical_operation,
                                      batch_size=batch_size, day_increment=day_increment, verbose=verbose)
    # if model_name == 'cnn_lstm_v2':
    #     model = build_cnn_lstm_v2_model(train, train_label, n_input=n_input, lstm_units=lstm_units,
    #                                     decoder_dense_units=decoder_dense_units, conv1d_filters=conv1d_filters,
    #                                     optimizer=optimizer, epochs=epochs, n_out=n_out, n_gap=n_gap)

    return model


def moving_window_predictions(daily_data: pd.DataFrame, model_name: str, model_type: str, lstm_units: int,
                              decoder_dense_units: int, epochs: int, statistical_operation: Dict, learning_rate: float,
                              beta_1: float, beta_2: float, epsilon: float, n_input: int, n_out: int, n_gap: int) -> \
        Tuple['np.ndarray', 'np.ndarray']:
    """
    Args:
        daily_data: data for train/validation.
        epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
                Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is
                not trained for a number of iterations given by epochs, but merely until the epoch of index epochs
                is reached.
        lstm_units: Positive integer, dimensionality of the output space.
        decoder_dense_units: Positive integer, dimensionality of the output space.
        learning_rate: A Tensor, floating point value, or a schedule that is a
                       tf.keras.optimizers.schedules.LearningRateSchedule, or a callable that takes no arguments and
                       returns the actual value to use, The learning rate.
        beta_1: A float value or a constant float tensor, or a callable that takes no arguments and returns the actual
                value to use. The exponential decay rate for the 1st moment estimates.
        beta_2: A float value or a constant float tensor, or a callable that takes no arguments and returns the actual
                value to use, The exponential decay rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper
                (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper.
        model_name: code for neural network architecture.
        model_type: indication of daily2weekly or weekly2weekly model.
        n_input: time steps of encoder in days.
        n_out: time steps of decoder in weeks.
        n_gap: time difference between end of encoder and begin of decoder in weeks.
        statistical_operation: specify the method of data aggregation from daily data to weekly data.
    Returns:
        ground truth and model predictions
    """

    predictions = []
    observations = []

    n_splits = train_validation_config['n_splits']
    max_train_size = train_validation_config['max_train_size']
    scale = 100000

    tscv = TimeSeriesSplit(n_splits=n_splits + (n_out - 1), test_size=7, max_train_size=max_train_size)

    for train_index, val_index in tscv.split(daily_data):
        if (train_index[-1] + n_out * 7 + 1) <= len(daily_data):
            val_index = np.arange(train_index[-1] + 1, train_index[-1] + n_out * 7 + 1)

            train = daily_data.iloc[train_index].values / scale
            val = daily_data.iloc[np.concatenate((train_index[-(n_gap * 7):], val_index))].values / scale

            val_label = val[:, 0]
            train_label = train[:, 0]

            val = np.array(np.split(val, len(val) / 7))
            train = np.array(np.split(train, len(train) / 7))

            model = model_training(model_name=model_name, model_type=model_type, train=train, train_label=train_label,
                                   lstm_units=lstm_units, decoder_dense_units=decoder_dense_units, epochs=epochs,
                                   n_input=n_input, n_out=n_out, n_gap=n_gap,
                                   statistical_operation=statistical_operation,
                                   learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

            p, o = evaluation_model(model, train, val, val_label, n_input, n_out=n_out, n_gap=n_gap,
                                    statistical_operation=statistical_operation)

            observations.append(o)
            predictions.append(p)

    observations = np.concatenate(observations, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    return observations, predictions


def training_process(daily_data: pd.DataFrame, epochs: Union[int, float], lstm_units: Union[int, float],
                     decoder_dense_units: Union[int, float], learning_rate: float, beta_1: float, beta_2: float,
                     epsilon: float, model_name: str, model_type: str, n_input: int, n_out: int, n_gap: int,
                     statistical_operation: Dict) -> float:
    """
    Args:
        daily_data: data for train/validation.
        epochs: Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
                Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is
                not trained for a number of iterations given by epochs, but merely until the epoch of index epochs
                is reached.
        lstm_units: Positive integer, dimensionality of the output space.
        decoder_dense_units: Positive integer, dimensionality of the output space.
        learning_rate: A Tensor, floating point value, or a schedule that is a
                       tf.keras.optimizers.schedules.LearningRateSchedule, or a callable that takes no arguments and
                       returns the actual value to use, The learning rate.
        beta_1: A float value or a constant float tensor, or a callable that takes no arguments and returns the actual
                value to use. The exponential decay rate for the 1st moment estimates.
        beta_2: A float value or a constant float tensor, or a callable that takes no arguments and returns the actual
                value to use, The exponential decay rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper
                (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper.
        model_name: code for neural network architecture.
        model_type: indication of daily2weekly or weekly2weekly model.
        n_input: time steps of encoder in days.
        n_out: time steps of decoder in weeks.
        n_gap: time difference between end of encoder and begin of decoder in weeks.
        statistical_operation: specify the method of data aggregation from daily data to weekly data.
    Returns:
        mean squared error of the moving window validation set.
    """

    lstm_units = int(lstm_units)
    decoder_dense_units = int(decoder_dense_units)
    epochs = int(epochs)

    observations, predictions = moving_window_predictions(daily_data, model_name, model_type, lstm_units=lstm_units,
                                                          decoder_dense_units=decoder_dense_units, epochs=epochs,
                                                          statistical_operation=statistical_operation,
                                                          learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2,
                                                          epsilon=epsilon, n_input=n_input, n_out=n_out, n_gap=n_gap)

    mse = mean_squared_error(observations, predictions)

    return -mse