import pytest

from ..ODEnlls import ODEnlls

def test_one_compound_A():
    x = ODEnlls()
    half = x._halfRxn('A')
    answer = (['A'], '(A)', [1.0])
    assert half == answer

def test_two_compounds_A_B():
    x = ODEnlls()
    half = x._halfRxn('A + B')
    answer = (['A', 'B'], '(A)*(B)', [1.0, 1.0])
    assert half == answer

def test_two_compounds_2B():
    x = ODEnlls()
    half = x._halfRxn('2*B')
    answer = (['B'], '((B**2.00)/2.00)', [2.0])

def test_three_compounds_A_2B():
    x = ODEnlls()
    half = x._halfRxn('A + 2*B')
    answer = (['A', 'B'], '(A)*((B**2.00)/2.00)', [1.0, 2.0])
    assert half == answer
