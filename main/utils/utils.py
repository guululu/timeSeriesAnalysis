from glob import glob
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import tensorflow as tf
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
        load_logs(optimizer, logs=previous_logs)

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


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):

    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    if len(seq.shape) == 3:
        seq = tf.reduce_sum(seq, axis=2)
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):

    '''
    x = tf.random.uniform((1, 3))
    temp = create_look_ahead_mask(x.shape[1])
    temp

    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[0., 1., 1.],
           [0., 0., 1.],
           [0., 0., 0.]], dtype=float32)>
    '''

    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch, num_heads, seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights
