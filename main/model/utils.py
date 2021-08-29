import numpy as np
from tensorflow.keras.models import Sequential
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_squared_log_error


def evaluate_forecasts(actual, predicted):

    msle = mean_squared_log_error(actual, np.clip(predicted, 0, 100))
    mse = mean_squared_error(actual, predicted)

    return msle, mse


def to_supervised(train, label_train, n_input, n_out=6, n_future=7):
    '''
    n_input: days
    n_out: measured in weeks
    n_future: measured in weeks
    '''

    # Multivariant input
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_start = in_end + 7 * n_future
        out_end = out_start + 7 * n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            # Univariant version
            '''
            x_input = data[in_start:in_end, 0]
             x_input = x_input.reshape((len(x_input), 1))
            '''
            X.append(data[in_start:in_end, :])
            y.append(np.array(np.split(label_train[out_start: out_end], n_out)).sum(axis=1))
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y)


def forecast(model, history, n_input):
    # flatten data
    data = np.array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n_feature] Multivariant input
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


def evaluation_model(model: Sequential, train, test, label_test, n_input, n_out=6, n_future=7):

    history = [x for x in train]

    predictions = list()
    observations = list()

    for i in range(len(test) - (n_out + n_future) + 1):  # because we want to predict up to 13 weeks <-- wtf...
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        observation = np.split(label_test[(i + n_future) * 7: (i + n_out + n_future) * 7], n_out)
        observations.append(np.array(observation).sum(axis=1))
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    predictions = np.array(predictions)[:, :, 0]
    observations = np.array(observations)

    return predictions, observations
