import pandas as pd
from os.path import dirname, join


def load_data(data_file_name):
    module_path = dirname(__file__)
    path = join(module_path, 'data', data_file_name)
    data = pd.read_csv(path)
    return data
