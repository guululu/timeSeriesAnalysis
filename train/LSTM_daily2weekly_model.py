from typing import Dict, Optional

from main.utils.utils import time_series_data_preparation
from main.model.LSTM_daily2weekly_architecture import build_lstm_v1_architecture, build_lstm_v2_architecture, \
    build_lstm_v3_architecture, build_cnn_lstm_v2_architecture


def build_lstm_v1_model(train, train_label, n_input, lstm_units, decoder_dense_units, optimizer,
                        epochs=35, n_out=6, n_gap=7, day_increment=7, batch_size=16, verbose=0,
                        statistical_operation: Dict = None):
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


def build_lstm_v2_model(train, train_label, n_input, lstm_units, decoder_dense_units, optimizer,
                        epochs=35, n_out=6, n_gap=7, day_increment=7, batch_size=16, verbose=0,
                        statistical_operation: Dict = None):
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


def build_lstm_v3_model(train, train_label, n_input, lstm_units, decoder_dense_units, optimizer,
                        epochs=35, n_out=6, n_gap=7, day_increment=7, batch_size=16, verbose=0,
                        statistical_operation: Optional[Dict] = None):
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


def build_cnn_lstm_v2_model(train, train_label, n_input, lstm_units, decoder_dense_units, conv1d_filters, optimizer,
                            epochs=35, n_out=6, n_gap=7, day_increment=7, batch_size=16, verbose=0,
                            statistical_operation: Optional[Dict] = None):
    # prepare data
    train_x, _, _, train_y_weekly = time_series_data_preparation(train, train_label, n_input,
                                                                 n_out=n_out, n_gap=n_gap,
                                                                 day_increment=day_increment,
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