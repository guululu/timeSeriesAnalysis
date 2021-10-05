from typing import Dict

from main.utils.utils import time_series_data_preparation
from main.model.LSTM_weekly2weekly_architecture import build_lstm_v2_architecture, build_lstm_feed_architecture, \
    build_cnn_lstm_v2_architecture, build_cnn_lstm_v2_luong_architecture, build_conv2d_lstm_v2_architecture


def build_lstm_v2_model(train, train_label, n_input, lstm_units, decoder_dense_units, optimizer,
                        epochs=35, n_out=6, n_gap=7, day_increment=7,
                        batch_size=16, verbose=0, statistical_operation: Dict = None):
    # prepare data
    _, train_x_weekly, _, _, train_y_weekly = time_series_data_preparation(train, train_label, n_input,
                                                                           n_out=n_out, n_gap=n_gap,
                                                                           day_increment=day_increment,
                                                                           statistical_operation=statistical_operation)

    # define parameters
    n_timesteps, n_features, n_outputs = train_x_weekly.shape[1], train_x_weekly.shape[2], train_y_weekly.shape[1]

    # define model
    model = build_lstm_v2_architecture(lstm_units, decoder_dense_units, n_timesteps, n_features, n_outputs)

    model.compile(loss='mse', optimizer=optimizer)
    # fit network
    model.fit({'weekly_inputs': train_x_weekly},
              {'outputs': train_y_weekly}, epochs=epochs, batch_size=batch_size, verbose=verbose,
              shuffle=True)

    return model


def build_lstm_feed_model(train, train_label, n_input, lstm_units, decoder_dense_units, optimizer,
                          epochs=35, n_out=6, n_gap=7, day_increment=7,
                          batch_size=16, verbose=0, statistical_operation: Dict = None):
    # prepare data
    _, train_x_weekly, _, _, train_y_weekly = time_series_data_preparation(train, train_label, n_input,
                                                                           n_out=n_out, n_gap=n_gap,
                                                                           day_increment=day_increment,
                                                                           statistical_operation=statistical_operation)

    # define parameters
    n_timesteps, n_features, n_outputs = train_x_weekly.shape[1], train_x_weekly.shape[2], train_y_weekly.shape[1]

    # define model
    model = build_lstm_feed_architecture(lstm_units, decoder_dense_units, n_timesteps, n_features, n_outputs)

    model.compile(loss='mlse', optimizer=optimizer)
    # fit network
    model.fit({'weekly_inputs': train_x_weekly},
              {'outputs': train_y_weekly}, epochs=epochs, batch_size=batch_size, verbose=verbose,
              shuffle=True)

    return model


def build_cnn_lstm_v2_model(train, train_label, n_input, lstm_units, decoder_dense_units, conv1d_filters, optimizer,
                            epochs=35, n_out=6, n_gap=7, day_increment=7,
                            batch_size=16, verbose=0, statistical_operation: Dict = None):
    # prepare data
    _, train_x_weekly, _, _, train_y_weekly = time_series_data_preparation(train, train_label, n_input,
                                                                           n_out=n_out, n_gap=n_gap,
                                                                           day_increment=day_increment,
                                                                           statistical_operation=statistical_operation)

    # define parameters
    n_timesteps, n_features, n_outputs = train_x_weekly.shape[1], train_x_weekly.shape[2], train_y_weekly.shape[1]

    # define model
    model = build_cnn_lstm_v2_architecture(lstm_units, decoder_dense_units, conv1d_filters,
                                           n_timesteps, n_features, n_outputs)

    model.compile(loss='mse', optimizer=optimizer)
    # fit network
    model.fit({'weekly_inputs': train_x_weekly},
              {'outputs': train_y_weekly}, epochs=epochs, batch_size=batch_size, verbose=verbose,
              shuffle=True)

    return model


def build_cnn_lstm_v2_luong_model(train, train_label, n_input, lstm_units, decoder_dense_units, conv1d_filters,
                                  optimizer, epochs=35, n_out=6, n_gap=7, day_increment=7,
                                  batch_size=16, verbose=0, statistical_operation: Dict = None):
    # prepare data
    _, train_x_weekly, _, _, train_y_weekly = time_series_data_preparation(train, train_label, n_input,
                                                                           n_out=n_out, n_gap=n_gap,
                                                                           day_increment=day_increment,
                                                                           statistical_operation=statistical_operation)

    # define parameters
    n_timesteps, n_features, n_outputs = train_x_weekly.shape[1], train_x_weekly.shape[2], train_y_weekly.shape[1]

    # define model
    model = build_cnn_lstm_v2_luong_architecture(lstm_units, decoder_dense_units, conv1d_filters,
                                                 n_timesteps, n_features, n_outputs)

    model.compile(loss='mse', optimizer=optimizer, )
    # fit network
    model.fit({'weekly_inputs': train_x_weekly}, {'outputs': train_y_weekly},
              epochs=epochs, batch_size=batch_size, verbose=verbose,
              shuffle=True)

    return model


def build_conv2d_lstm_v2_model(train, train_label, n_input, decoder_dense_units, filters, optimizer,
                               epochs=35, n_out=6, n_gap=7, day_increment=7,
                               batch_size=16, verbose=0, statistical_operation: Dict = None):
    # prepare data
    _, train_x_weekly, _, _, train_y_weekly = time_series_data_preparation(train, train_label, n_input,
                                                                           n_out=n_out, n_gap=n_gap,
                                                                           day_increment=day_increment,
                                                                           statistical_operation=statistical_operation)

    # define parameters

    train_x_weekly = train_x_weekly.reshape(train_x_weekly.shape[0], 2, 1, 4, train_x_weekly.shape[-1])

    n_features, n_outputs = train_x_weekly.shape[-1], train_y_weekly.shape[1]

    # define model
    model = build_conv2d_lstm_v2_architecture(decoder_dense_units, filters, n_features, n_outputs)

    model.compile(loss='mse', optimizer=optimizer)
    # fit network
    model.fit({'weekly_inputs': train_x_weekly},
              {'outputs': train_y_weekly}, epochs=epochs, batch_size=batch_size, verbose=verbose,
              shuffle=True)

    return model