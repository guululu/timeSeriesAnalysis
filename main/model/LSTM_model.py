import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate


# def lstm_basic_model(history_points: int, features: int):
#
#     tf.random.set_seed(20)
#     np.random.seed(10)
#     lstm_input = Input(shape=(history_points, features), name='lstm_input')
#
#     inputs = LSTM(30, name='first_layer')(lstm_input)
#     inputs = Dense(10)(inputs)
#     inputs = Dense(1)(inputs)
#     output = Activation('linear', name='output')(inputs)
#
#     model = Model(inputs=lstm_input, outputs=output)
#
#     return model
#
#
# def lstm_seq_model(history_points: int, features: int):
#
#     tf.random.set_seed(20)
#     np.random.seed(10)
#     lstm_input = Input(shape=(history_points, features), name='lstm_input')
#
#     inputs = LSTM(24, name='first_layer', return_sequences=True)(lstm_input)
#     inputs = LSTM(72, name='second_layer')(inputs)
#     inputs = Dense(144, activation='relu')(inputs)
#     inputs = Dense(36)(inputs)
#     inputs = Dense(1)(inputs)
#     output = Activation('linear', name='output')(inputs)
#
#     model = Model(inputs=lstm_input, outputs=output)
#
#     return model