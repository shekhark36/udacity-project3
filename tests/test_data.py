import sys

from sklearn.preprocessing import OneHotEncoder, StandardScaler
sys.path.append('../udacity-project3')

import os
import pandas as pd
from starter.code.data import process_data

def test_column_integrity(data: pd.DataFrame, expected_columns: list):
    
    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_columns) == list(these_columns)

def test_shape(data, expected_columns):
    assert data.shape[0] > 1000
    assert data.shape[1] == len(expected_columns)

def test_process_data(data, categorical_features, target_label):
    _, _, encoder, scaler, lb = process_data(data, categorical_features, 
                                            target_label, training=True, 
                                            encoder=None, scaler=None, 
                                            lb=None)
    
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(scaler, StandardScaler)
