import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, TimeDistributed, Dense, RepeatVector, Concatenate, Lambda, Average, \
    Dropout, Conv1D, Flatten, MaxPooling1D, Dot, Activation


def build_lstm_v1_architecture(lstm_units, decoder_dense_units, n_input, n_feature, n_out):
    '''
    Args:
        lstm_units: Positive integer, dimensionality of the output space.
        decoder_dense_units: Positive integer, dimensionality of the output space.
        n_input: time steps of encoder in days.
        n_feature: number of features.
        n_out: time steps of decoder in weeks.

    Returns:
        A tensorflow model architecture
    '''

    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    random.seed(42)
    np.random.seed(42)

    # define model
    main_inputs = Input(shape=(n_input, n_feature), name='daily_inputs')
    x, state_h, state_c = LSTM(lstm_units, activation='tanh', return_state=True, dropout=0.1,
                               recurrent_dropout=0.1)(main_inputs)

    weekly_inputs = Input(shape=(n_out, n_feature), name='weekly_inputs')

    y = LSTM(lstm_units, activation='tanh', return_sequences=True, dropout=0.1,
             recurrent_dropout=0.1)(weekly_inputs, initial_state=[state_h, state_c])

    y = TimeDistributed(Dense(decoder_dense_units, activation='relu'))(y)
    outputs = TimeDistributed(Dense(1), name='outputs')(y)

    model = Model(inputs=[main_inputs, weekly_inputs], outputs=outputs)

    return model


def build_lstm_v2_architecture(lstm_units: int, decoder_dense_units: int, n_input: int, n_feature: int, n_out: int):

    '''
    Args:
        lstm_units: Positive integer, dimensionality of the output space.
        decoder_dense_units: Positive integer, dimensionality of the output space.
        n_input: time steps of encoder in days.
        n_feature: number of features.
        n_out: time steps of decoder in weeks.

    Returns:
        A tensorflow model architecture
    '''

    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    random.seed(42)
    np.random.seed(42)

    # define model
    main_inputs = Input(shape=(n_input, n_feature), name='daily_inputs')
    x, state_h, state_c = LSTM(lstm_units, activation='tanh', return_state=True, dropout=0.1,
                               recurrent_dropout=0.1)(main_inputs)

    decoder_input = RepeatVector(n_out)(x)  # Repeatvector(n_out)(state_h)

    y = LSTM(lstm_units, activation='tanh', return_sequences=True, dropout=0.1,
             recurrent_dropout=0.1)(decoder_input, initial_state=[state_h, state_c])

    y = TimeDistributed(Dense(decoder_dense_units, activation='relu'))(y)
    outputs = TimeDistributed(Dense(1), name='outputs')(y)

    model = Model(inputs=main_inputs, outputs=outputs)

    return model


def build_lstm_v3_architecture(lstm_units: int, decoder_dense_units: int, n_input: int, n_feature: int, n_out: int):
    '''
    Args:
        lstm_units: Positive integer, dimensionality of the output space.
        decoder_dense_units: Positive integer, dimensionality of the output space.
        n_input: time steps of encoder in days.
        n_feature: number of features.
        n_out: time steps of decoder in weeks.

    Returns:
        A tensorflow model architecture
    '''

    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    random.seed(42)
    np.random.seed(42)

    # define model
    main_inputs = Input(shape=(n_input, n_feature), name='daily_inputs')
    x, state_h, state_c = LSTM(lstm_units, activation='tanh', return_state=True, dropout=0.1,
                               recurrent_dropout=0.1)(main_inputs)

    decoder_input = RepeatVector(n_out)(x)  # Repeatvector(n_out)(state_h)
    weekly_inputs = Input(shape=(n_out, n_feature), name='weekly_inputs')
    decoder_input = Concatenate(axis=2)([decoder_input, weekly_inputs])

    y = LSTM(lstm_units, activation='tanh', return_sequences=True, dropout=0.1,
             recurrent_dropout=0.1)(decoder_input, initial_state=[state_h, state_c])
    y = TimeDistributed(Dense(decoder_dense_units, activation='relu'))(y)

    outputs = TimeDistributed(Dense(1), name='outputs')(y)

    model = Model(inputs=[main_inputs, weekly_inputs], outputs=outputs)

    return model


def build_lstm_feed_architecture(lstm_units: int, decoder_dense_units: int, n_input: int, n_feature: int, n_out: int):
    '''
    Args:
        lstm_units: Positive integer, dimensionality of the output space.
        decoder_dense_units: Positive integer, dimensionality of the output space.
        n_input: time steps of encoder in days.
        n_feature: number of features.
        n_out: time steps of decoder in weeks.

    Returns:
        A tensorflow model architecture
    '''

    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    random.seed(42)
    np.random.seed(42)

    # define model
    main_inputs = Input(shape=(n_input, n_feature), name='daily_inputs')
    x, state_h, state_c = LSTM(lstm_units, activation='tanh', return_state=True, dropout=0.1,
                               recurrent_dropout=0.1, name='LSTM_encoder')(main_inputs)

    input_context = tf.expand_dims(x, axis=1)

    decoder_lstm = LSTM(lstm_units, activation='tanh', return_state=True, return_sequences=True,
                        dropout=0.1, recurrent_dropout=0.1, name='LSTM_decoder')
    decoder_dense = Dense(decoder_dense_units, activation='relu', name='decoder_dense')
    output_dense = Dense(1, activation='relu', name='output_dense')

    input = input_context
    states = [state_h, state_c]
    all_outputs = []

    for it in range(n_out):
        # step 1. feed concatenated inputs into decoder with concatenated inputs defined
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(input, initial_state=states)

        input = decoder_outputs  # the final output will be feed into the next LSTM cell
        all_outputs.append(input)
        input = Average()(all_outputs + [input_context])
        states = [decoder_state_h, decoder_state_c]

    y = Lambda(lambda x: K.concatenate(x, axis=1), name='daily_outputs')(all_outputs)
    y = TimeDistributed(decoder_dense)(y)

    outputs = TimeDistributed(output_dense, name='outputs')(y)

    model = Model(inputs=main_inputs, outputs=outputs)

    return model


def build_cnn_lstm_v2_architecture(lstm_units: int, decoder_dense_units: int, conv1d_filters: int,
                                   n_input: int, n_feature: int, n_out: int):
    '''
    A CNN-LSTM seq2seq model by replacing the LSTM in encoder with Con1D layers

    Args:
        lstm_units: Positive integer, dimensionality of the output space.
        decoder_dense_units: Positive integer, dimensionality of the output space.
        conv1d_filters: Integer, the dimensionality of the output space
                        (i.e. the number of output filters in the convolution).
        n_input: time steps of encoder in days.
        n_feature: number of features.
        n_out: time steps of decoder in weeks.

    Returns:
        A tensorflow model architecture
    '''


    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    random.seed(42)
    np.random.seed(42)

    # define model
    main_inputs = Input(shape=(n_input, n_feature), name='daily_inputs')
    x = Conv1D(filters=conv1d_filters, kernel_size=3, activation='relu', padding='same',  # 加權平均
               input_shape=(n_input, n_feature))(main_inputs)
    x = Dropout(0.3)(x)
    x = Conv1D(filters=conv1d_filters, kernel_size=3, activation='relu', padding='same',  # 加權平均
               input_shape=(n_input, n_feature))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    decoder_input = RepeatVector(n_out)(x)  # Repeatvector(n_out)(state_h)

    y = LSTM(lstm_units, activation='tanh', return_sequences=True, dropout=0.1,
             recurrent_dropout=0.1)(decoder_input)

    y = TimeDistributed(Dense(decoder_dense_units, activation='relu'))(y)
    outputs = TimeDistributed(Dense(1), name='outputs')(y)

    model = Model(inputs=main_inputs, outputs=outputs)

    return model


def build_cnn_lstm_v2_luong_architecture(lstm_units: int, decoder_dense_units: int, conv1d_filters: int,
                                         n_input: int, n_feature: int, n_out: int):
    '''
    A CNN-LSTM seq2seq model by replacing the LSTM in encoder with Con1D layers

    Args:
        lstm_units: Positive integer, dimensionality of the output space.
        decoder_dense_units: Positive integer, dimensionality of the output space.
        conv1d_filters: Integer, the dimensionality of the output space
                        (i.e. the number of output filters in the convolution).
        n_input: time steps of encoder in days.
        n_feature: number of features.
        n_out: time steps of decoder in weeks.

    Returns:
        A tensorflow model architecture
    '''

    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    random.seed(42)
    np.random.seed(42)

    decoder_lstm = LSTM(lstm_units, activation='tanh', return_sequences=True, name='LSTM_decoder_1', dropout=0.1,
                        recurrent_dropout=0.1)
    decoder_dense = Dense(decoder_dense_units, activation='relu', name='decoder_dense')
    output_dense = Dense(1, activation='relu', name='output_dense')

    # define model
    main_inputs = Input(shape=(n_input, n_feature), name='daily_inputs')
    x_days = Conv1D(filters=conv1d_filters, kernel_size=3, activation='relu', padding='same',  # 加權平均
                    input_shape=(n_input, n_feature), name='Conv1D_encoder_a')(main_inputs)
    x_days = Dropout(0.3)(x_days)
    x_days = Conv1D(filters=lstm_units, kernel_size=3, activation='relu', padding='same', name='Conv1D_encoder_b')(x_days)

    encoder_stack_h = Conv1D(filters=lstm_units, kernel_size=3, activation='relu',
                             padding='same', name='Conv1D_encoder_c')(x_days)

    x_days = MaxPooling1D(pool_size=4)(encoder_stack_h)
    x_days = Flatten()(x_days)
    decoder_input = RepeatVector(n_out)(x_days)  # Repeatvector(n_out)(state_h)

    decoder_stack_h = decoder_lstm(decoder_input)

    ### Attention ###

    attention = Dot(axes=[2, 2])([decoder_stack_h, encoder_stack_h])
    attention = Activation('softmax')(attention)

    context = Dot(axes=[2, 1])([attention, encoder_stack_h])

    attention_hidden_state = Activation('tanh')(Concatenate(axis=2)([context, decoder_stack_h]))

    y = TimeDistributed(decoder_dense)(attention_hidden_state )
    outputs = TimeDistributed(output_dense, name='outputs')(y)

    model = Model(inputs=main_inputs, outputs=outputs)

    return model
