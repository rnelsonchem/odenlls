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

def test_function_string():
    '''This test is surrogate for testing the read_rxns method. There are
    other things to be tested in this function as well.
    '''
    x = ODEnlls()
    x.read_rxns(os.path.join(current_directory, 'sample_rxns.txt'))
    answer = """def f(y, t, k1, k2, k3, k4, k5, k6):
    return [
        1.00*(-1*k1*(y[0]) + k2*(y[1])) + 1.00*(-1*k3*(y[0]) + k4*(y[2])) + 1.00*(-1*k5*(y[0]) + k6*(y[3])),
        1.00*(k1*(y[0]) + -1*k2*(y[1])),
        1.00*(k3*(y[0]) + -1*k4*(y[2])),
        1.00*(k5*(y[0]) + -1*k6*(y[3])),
        ]"""
    assert x._function_string == answer
