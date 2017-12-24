import os

import pytest
import numpy as np
import pandas as pd

from ..ODEnlls import ODEnlls

path_to_current_file = os.path.realpath(__file__)
current_directory = os.path.dirname(path_to_current_file)

def test_read_data():
    x = ODEnlls()
    x.read_data(os.path.join(current_directory, 'sample_data.txt'))
    answer = pd.read_pickle(os.path.join(current_directory, 
                                        'sample_data.pkl'))
    assert np.alltrue(answer == x.data)
