import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, TimeDistributed, Dense, RepeatVector, Add, Concatenate
from tensorflow.keras.optimizers import Adam

from main.utils.utils import to_supervised


def build_lstm_v1_model(train, train_label, n_input, lstm_filter, dense_filter_decoder,
                        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, epochs=35, n_out=6, n_gap=7):
    # prepare data
    train_x, train_y, train_x_weekly = to_supervised(train, train_label, n_input, n_out=n_out, n_gap=n_gap)

    # define parameters
    verbose, batch_size = 0, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    tf.random.set_seed(42)

    os.environ['PYTHONHASHSEED'] = '42'

    random.seed(42)
    np.random.seed(42)

    # define model

    main_inputs = Input(shape=(n_timesteps, n_features), name='main_inputs')
    x, state_h, state_c = LSTM(lstm_filter, activation='relu', return_state=True)(main_inputs)

    _, _, dim = train_x_weekly.shape

    weekly_inputs = Input(shape=(n_outputs, dim), name='weekly_inputs')

    y = LSTM(lstm_filter, activation='relu', return_sequences=True)(weekly_inputs, initial_state=[state_h, state_c])

    y = TimeDistributed(Dense(dense_filter_decoder, activation='relu'))(y)
    outputs = TimeDistributed(Dense(1), name='outputs')(y)

    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    model = Model(inputs=[main_inputs, weekly_inputs], outputs=outputs)
    model.compile(loss='mse', optimizer=optimizer)
    # fit network
    model.fit({'main_inputs': train_x, 'weekly_inputs': train_x_weekly},
              {'outputs': train_y}, epochs=epochs, batch_size=batch_size, verbose=verbose,
              shuffle=True)

    return model


def build_lstm_v2_model(train, train_label, n_input, lstm_filter, dense_filter_decoder,
                        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, epochs=30, n_out=6, n_gap=7):
    # prepare data
    train_x, train_y, train_x_weekly = to_supervised(train, train_label, n_input, n_out=n_out, n_gap=n_gap)

    # define parameters
    verbose, batch_size = 0, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    tf.random.set_seed(42)

    os.environ['PYTHONHASHSEED'] = '42'

    random.seed(42)
    np.random.seed(42)

    # define model

    main_inputs = Input(shape=(n_timesteps, n_features), name='main_inputs')
    x, state_h, state_c = LSTM(lstm_filter, activation='relu', return_state=True, dropout=0.3,
                               recurrent_dropout=0.1)(main_inputs)

    decoder_input = RepeatVector(n_out)(x)  # Repeatvector(n_out)(state_h)

    y = LSTM(lstm_filter, activation='relu', return_sequences=True, dropout=0.3,
             recurrent_dropout=0.1)(decoder_input, initial_state=[state_h, state_c])

    _, _, dim = train_x_weekly.shape
    weekly_inputs = Input(shape=(n_outputs, dim), name='weekly_inputs')  # 無作用

    y = TimeDistributed(Dense(dense_filter_decoder, activation='relu'))(y)
    outputs = TimeDistributed(Dense(1), name='outputs')(y)

    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    model = Model(inputs=[main_inputs, weekly_inputs], outputs=outputs)
    model.compile(loss='mse', optimizer=optimizer)
    # fit network
    model.fit({'main_inputs': train_x, 'weekly_inputs': train_x_weekly},
              {'outputs': train_y}, epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model


def build_lstm_v3_model(train, train_label, n_input, lstm_filter, dense_filter_decoder,
                        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, epochs=30, n_out=6, n_gap=7):
    # prepare data
    train_x, train_y, train_x_weekly = to_supervised(train, train_label, n_input, n_out=n_out, n_gap=n_gap)

    # define parameters
    verbose, batch_size = 0, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    tf.random.set_seed(42)

    os.environ['PYTHONHASHSEED'] = '42'

    random.seed(42)
    np.random.seed(42)

    # define model

    main_inputs = Input(shape=(n_timesteps, n_features), name='main_inputs')
    x, state_h, state_c = LSTM(lstm_filter, activation='relu', return_state=True, dropout=0.3,
                               recurrent_dropout=0.1)(main_inputs)

    decoder_input = RepeatVector(n_out)(x)  # Repeatvector(n_out)(state_h)

    # use the last hidden state in encoder LSTM as input
    y_decoder = LSTM(lstm_filter, activation='relu', return_sequences=True, dropout=0.3,
                     recurrent_dropout=0.1)(decoder_input, initial_state=[state_h, state_c])
    y_decoder = TimeDistributed(Dense(dense_filter_decoder, activation='relu'))(y_decoder)

    # user four weeks weekly average as input
    _, _, dim = train_x_weekly.shape
    weekly_inputs = Input(shape=(n_outputs, dim), name='weekly_inputs')
    y_weekly = LSTM(lstm_filter, activation='relu', return_sequences=True, dropout=0.3,
                    recurrent_dropout=0.1)(weekly_inputs, initial_state=[state_h, state_c])
    y_weekly = TimeDistributed(Dense(dense_filter_decoder, activation='relu'))(y_weekly)

    # concatenate the two results along the feature direction (axis=2)
    # axis 0: data row direction
    # axis 1: data time direction
    y = Concatenate(axis=2)([y_decoder, y_weekly])

    outputs = TimeDistributed(Dense(1), name='outputs')(y)

    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    model = Model(inputs=[main_inputs, weekly_inputs], outputs=outputs)
    model.compile(loss='mse', optimizer=optimizer)
    # fit network
    model.fit({'main_inputs': train_x, 'weekly_inputs': train_x_weekly},
              {'outputs': train_y}, epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model
