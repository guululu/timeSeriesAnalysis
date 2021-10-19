"""
This script is for searching optimized hyperparameters

In the command line interface (CLI), the code for execution is

python -m --model_name=<> --model_type=<>

For model_type daily2weekly, the following architectures are supported:
1. lstm_v1
2. lstm_v2
3. lstm_v3

The model details can be found in main/model/LSTM_daily2weekly_architecture.py

while for model_type weekly2weekly, the following architures are supported:
1. lstm_v2
"""


import argparse
from functools import partial
from collections import OrderedDict

from main.utils.data import raw_data_preparation
from train.training_process import training_process
from main.settings import holiday, requirements, features
from main.utils.utils import optimization_process


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', dest="MODEL_NAME", type=str, help='model architecture name')
    parser.add_argument('--model_type', dest='MODEL_TYPE', type=str, help='input time segmentation, daily or weekly')

    args = parser.parse_args()

    model_name = args.MODEL_NAME
    model_type = args.MODEL_TYPE

    # prediction for n_out weeks
    n_out = requirements['n_out']
    # n_gap weeks gap
    n_gap = requirements['n_gap']
    # n_input days input
    n_input = requirements['n_input']

    df = raw_data_preparation()

    main_feature = features['main_feature']
    additional_features = features['additional_features']

    df[f"{main_feature}_diff"] = df[f"{main_feature}"].diff()

    daily_data = df[[f'{main_feature}'] + additional_features + [f'{main_feature}_diff']]

    statistical_operation = OrderedDict()

    statistical_operation[0] = ['sum']
    statistical_operation[1] = ['sum']
    statistical_operation[2] = ['sum']
    statistical_operation[3] = ['sum']

    pbounds = {'epochs': (5, 20),
               'lstm_units': (48, 130),
               'decoder_dense_units': (8, 20),
               'learning_rate': (0.0001, 0.003),
               'beta_1': (0.5, 0.95),
               'beta_2': (0.7, 0.9999),
               'epsilon': (0.00001, 0.01)}

    training_process_fn = partial(training_process, daily_data=daily_data, model_name=model_name,
                                  model_type=model_type, n_out=n_out, n_gap=n_gap, n_input=n_input,
                                  statistical_operation=statistical_operation)

    optimized_parameters = optimization_process(training_process_fn, pbounds, model_name=model_name,
                                                model_type=model_type)
