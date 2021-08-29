import pandas as pd

from main.io.metadata_definition import set_holiday, set_training_config

globals().update(set_holiday())
globals().update(set_training_config())

# Transform time from string into Timestamp object
holiday = [pd.Timestamp(t) for t in holiday]