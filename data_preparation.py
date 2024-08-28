import pandas as pd
from sklearn.model_selection import train_test_split


cols_features = [
    'f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11'
]

col_treatment = 'treatment'


data = pd.read_csv('data/full_data.csv', index_col=[0])

train_index, test_index = train_test_split(data.index, test_size=0.2, random_state=42)

_sample_treated_size = 100000
_sample_control_size = 100000

col_treatment = 'treatment'

train_sample_index = data.loc[train_index, [col_treatment]]\
    .query(f'{col_treatment} == 1').sample(_sample_treated_size).index

train_sample_index = train_sample_index.union(data.loc[train_index, [col_treatment]]\
                                              .query(f'{col_treatment} == 0')\
                                                .sample(_sample_control_size).index)

test_sample_index = data.loc[test_index, [col_treatment]]\
    .query(f'{col_treatment} == 1').sample(_sample_treated_size).index

test_sample_index  = test_sample_index.union(data.loc[test_index, [col_treatment]]\
                                             .query(f'{col_treatment} == 0')\
                                                .sample(_sample_control_size).index)

