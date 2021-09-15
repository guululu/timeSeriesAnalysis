import os
import yaml
from typing import Dict

from main.io.path_definition import get_file


def _load_yaml(file):

    with open(file, 'r') as f:
        loaded_yaml = yaml.full_load(f)
    return loaded_yaml


def set_holiday() -> Dict:

    """
    Get the parameters for Bayesian optimization for hyperparameters

    Returns:
        a dictionary containing the hyper parameters for Bayesian optimization process
    """

    return _load_yaml(get_file(os.path.join('config', 'holiday.yml')))


def set_training_config() -> Dict:

    """
    Get the parameters for Bayesian optimization for hyperparameters

    Returns:
        a dictionary containing the hyper parameters for Bayesian optimization process
    """

    return _load_yaml(get_file(os.path.join('config', 'training_config.yml')))