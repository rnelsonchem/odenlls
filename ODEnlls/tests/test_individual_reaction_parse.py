import pytest

from ..ODEnlls import ODEnlls

def test_irrev_A_to_B():
    x = ODEnlls()
    rxn = x._rxnRate('A -> B')
    answer = (1, ['A', 'B'], ['1.00*(-1*k1*(A))', '1.00*(k1*(A))'])
    assert rxn == answer

def test_irrev_A_to_B_C():
    x = ODEnlls()
    rxn = x._rxnRate('A -> B + C')
    answer = (1, ['A', 'B', 'C'], 
            ['1.00*(-1*k1*(A))', '1.00*(k1*(A))', '1.00*(k1*(A))'])
    assert rxn == answer

def test_irrev_A_B_to_C():
    x = ODEnlls()
    rxn = x._rxnRate('A + B -> C')
    answer = (1, ['A', 'B', 'C'],
        ['1.00*(-1*k1*(A)*(B))', '1.00*(-1*k1*(A)*(B))', 
            '1.00*(k1*(A)*(B))']
        )
    assert rxn == answer

def test_irrev_A_to_2B():
    x = ODEnlls()
    rxn = x._rxnRate('A -> 2*B')
    answer = (1, ['A', 'B'], ['1.00*(-1*k1*(A))', '2.00*(k1*(A))'])
    assert rxn == answer

def test_irrev_2A_to_B():
    x = ODEnlls()
    rxn = x._rxnRate('2*A -> B')
    answer = (1, ['A', 'B'],
        ['2.00*(-1*k1*((A**2.00)/2.00))', '1.00*(k1*((A**2.00)/2.00))'])
    assert rxn == answer

def test_rev_A_to_B():
    x = ODEnlls()
    rxn = x._rxnRate('A = B')
    answer = (2, ['A', 'B'], 
            ['1.00*(-1*k1*(A) + k2*(B))', '1.00*(k1*(A) + -1*k2*(B))']) 
    assert rxn == answer

def test_rev_A_to_B_C():
    x = ODEnlls()
    rxn = x._rxnRate('A = B + C')
    answer = (2, ['A', 'B', 'C'],
         ['1.00*(-1*k1*(A) + k2*(B)*(C))',
          '1.00*(k1*(A) + -1*k2*(B)*(C))',
          '1.00*(k1*(A) + -1*k2*(B)*(C))']
         ) 
    assert rxn == answer

def test_rev_A_B_to_C():
    x = ODEnlls()
    rxn = x._rxnRate('A + B = C')
    answer = (2, ['A', 'B', 'C'],
         ['1.00*(-1*k1*(A)*(B) + k2*(C))',
          '1.00*(-1*k1*(A)*(B) + k2*(C))',
          '1.00*(k1*(A)*(B) + -1*k2*(C))']
         ) 
    assert rxn == answer

def test_rev_A_to_2B():
    x = ODEnlls()
    rxn = x._rxnRate('A = 2*B')
    answer = (2, ['A', 'B'],
         ['1.00*(-1*k1*(A) + k2*((B**2.00)/2.00))',
          '2.00*(k1*(A) + -1*k2*((B**2.00)/2.00))']
         ) 
    assert rxn == answer

def test_rev_2A_to_B():
    x = ODEnlls()
    rxn = x._rxnRate('2*A = B')
    answer = (2, ['A', 'B'],
         ['2.00*(-1*k1*((A**2.00)/2.00) + k2*(B))',
          '1.00*(k1*((A**2.00)/2.00) + -1*k2*(B))']
         )
    assert rxn == answer
