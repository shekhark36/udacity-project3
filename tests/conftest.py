import sys
sys.path.append('../udacity-project3')

import pytest
import pandas as pd

@pytest.fixture(scope='session')
def data(input_data_path="starter/data/cleaned_census_data.csv"):
    data = pd.read_csv(input_data_path)
    return data

@pytest.fixture(scope="session")
def expected_columns():
    return ['age',
            'workclass',
            'fnlgt',
            'education',
            'education_num',
            'marital_status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital_gain',
            'capital_loss',
            'hours_per_week',
            'native_country',
            'salary']

@pytest.fixture(scope='session')
def categorical_features():
    return [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

@pytest.fixture(scope='session')
def target_label():
    return 'salary'
