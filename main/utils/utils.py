from glob import glob
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from main.settings import bayesianOptimization
from main.io.path_definition import get_project_dir


def optimization_process(fn, pbounds: Dict, model_name: str, model_type: str) -> Tuple[Dict, np.ndarray]:

    """
    Bayesian optimization process interface. Returns hyperparameters of machine learning algorithms and the
    corresponding out-of-fold (oof) predictions. The progress will be saved into a json file.

    Args:
        fn: functional that will be optimized
        pbounds: a dictionary having the boundary of parameters of fn

    Returns:
        A tuple of dictionary containing optimized hyperparameters and oof-predictions
    """

    optimizer = BayesianOptimization(
        f=fn,
        pbounds=pbounds,
        random_state=1)

    export_form = datetime.now().strftime("%Y%m%d-%H")

    logs = f"{get_project_dir()}/data/optimization/{model_type}_{model_name}_logs_{export_form}.json"

    previous_logs = glob(f"{get_project_dir()}/data/optimization/{model_type}_{model_name}_logs_*.json")

    if previous_logs:
        load_logs(optimizer, logs=[logs])

    logger = JSONLogger(path=logs)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        **bayesianOptimization
    )
    optimized_parameters = optimizer.max['params']

    return optimized_parameters


def to_supervised(train, train_label, n_input: int, n_out: int, n_gap: int, day_increment: int = 7,
                  statistical_operation: Dict = None):
    '''
    Args:
        n_input: days
        n_out: measured in weeks
        n_gap: measured in weeks
        statistical_operation: used only for weekly2weekly model
    '''

    X, y, X_weekly, y_weekly = list(), list(), list(), list()

    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))

    data_record = {}

    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data) - (n_out + n_gap) * 7):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_start = in_end + 7 * n_gap
        out_end = out_start + 7 * n_out
        # ensure we have enough data for this instance

        feature_data_weekly = []

        if out_end <= len(data):

            data_record[_] = {'in_start': in_start, 'in_end': in_end, 'out_start': out_start, 'out_end': out_end}

            target_data = train_label[out_start: out_end]
            target_data_weekly = np.array(np.split(target_data, n_out)).sum(axis=1)
            if statistical_operation:
                for col_index, operations in statistical_operation.items():
                    daily_data = data[in_start: in_end, col_index]
                    # feature_data_daily.append(np.expand_dims(daily_data, 1))
                    for operation in operations:
                        statistical_feature = eval(f'np.nan{operation}(np.array(np.split(daily_data, n_input // 7)), '
                                                   f'axis=1, keepdims=True)')
                        if np.isnan(np.sum(statistical_feature)):
                            statistical_feature = np.nan_to_num(statistical_feature)
                        feature_data_weekly.append(statistical_feature)

            if statistical_operation:
                feature_data_weekly = np.concatenate(feature_data_weekly, axis=1)
            feature_data_daily = data[in_start: in_end]  #np.concatenate(feature_data_daily, axis=1)
            X.append(feature_data_daily)
            y.append(target_data)
            X_weekly.append(feature_data_weekly)
            y_weekly.append(target_data_weekly)
        else:
            break

        # add another week
        in_start += day_increment

    results = {'X': np.array(X), 'X_weekly': np.array(X_weekly),
               'y': np.array(y), 'y_weekly': np.array(y_weekly)}

    return results


def time_series_data_preparation(train, train_label, n_input: int, n_out: int, n_gap: int, day_increment: int,
                                 statistical_operation: Optional[Dict] = None):
    # prepare data
    data_dict = to_supervised(train, train_label, n_input, n_out=n_out,
                              n_gap=n_gap, day_increment=day_increment, statistical_operation=statistical_operation)

    train_x = data_dict['X']
    train_y = data_dict['y']
    train_x_weekly = data_dict['X_weekly']
    train_y_weekly = data_dict['y_weekly']

    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    train_y_weekly = train_y_weekly.reshape((train_y_weekly.shape[0], train_y_weekly.shape[1], 1))

    return train_x, train_x_weekly, train_y, train_y_weekly
