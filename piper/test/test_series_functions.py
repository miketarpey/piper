from piper.series_functions import ratio
import pandas as pd
import numpy as np

# test_dividing_numbers {{{1
def test_dividing_numbers():
    ''' '''

    exp = 1
    got = ratio(2, 2)

    assert exp == got


# test_dividing_numbers_by_zero {{{1
def test_dividing_numbers_by_zero():
    ''' '''

    exp = np.inf
    got = ratio(2, 0)

    assert exp == got


# test_dividing_numbers_floats {{{1
def test_dividing_numbers_floats():
    ''' '''

    exp = 1.0
    got = ratio(2.0, 2.0)

    assert exp == got


# test_dividing_numbers_float_percent {{{1
def test_dividing_numbers_float_percent():
    ''' '''

    exp = 100.0
    got = ratio(2.0, 2.0, percent=True)

    assert exp == got


# test_dividing_numbers_float_percent_with_round {{{1
def test_dividing_numbers_float_percent_with_round():
    ''' '''

    exp = 100.0000
    got = ratio(2.0, 2.0, percent=True, precision=4)

    assert exp == got

    exp = 50.00
    got = ratio(1.0, 2.0, percent=True, precision=2)

    assert exp == got


# test_dividing_numbers_int_percent_with_round {{{1
def test_dividing_numbers_int_percent_with_round():
    ''' '''

    exp = 100.0000
    got = ratio(2, 2, percent=True, precision=4)

    assert exp == got

    exp = 50.00
    got = ratio(1, 2, percent=True, precision=2)

    assert exp == got


# test_dividing_numbers_percent_with_format {{{1
def test_dividing_numbers_percent_with_format():
    ''' '''

    exp = '100.0%'
    got = ratio(2.0, 2.0, percent=True, format=True)

    assert exp == got


# test_dividing_numbers_percent_with_precision_format {{{1
def test_dividing_numbers_percent_with_precision_format():
    ''' '''

    exp = '66.66%'
    got = ratio(1.3333, 2.0, percent=True,
                precision=2, format=True)

    assert exp == got


# test_dividing_by_two_series {{{1
def test_dividing_by_two_series():
    ''' '''
    s1 = pd.Series([10, 20, 30])
    s2 = pd.Series([1, 2, 3])

    exp = pd.Series([10, 10, 10], dtype=float)

    got = ratio(s1, s2)

    assert exp.equals(got)


# test_dividing_by_two_series_with_zero_denominator {{{1
def test_dividing_by_two_series_with_zero_denominator():
    ''' '''
    s1 = pd.Series([10, 20, 30])
    s2 = pd.Series([1, 0, 3])

    exp = pd.Series([10, np.inf, 10], dtype=float)

    got = ratio(s1, s2)

    assert exp.equals(got)


# test_dividing_by_two_series_with_decimals {{{1
def test_dividing_by_two_series_with_decimals():
    ''' '''
    s1 = pd.Series([10, 20, 30])
    s2 = pd.Series([1.3, 5.4, 3])

    exp = (s1 / s2).round(2)

    got = ratio(s1, s2)

    assert exp.equals(got)


# test_dividing_by_two_series_with_rounding {{{1
def test_dividing_by_two_series_with_rounding():
    ''' '''
    s1 = pd.Series([10, 20, 30])
    s2 = pd.Series([1.3, 5.4, 3])

    exp = (s1 / s2).round(2)
    got = ratio(s1, s2, precision=2)
    assert exp.equals(got)

    exp = (s1 / s2).round(4)
    got = ratio(s1, s2, precision=4)
    assert exp.equals(got)


# test_dividing_by_two_series_with_format {{{1
def test_dividing_by_two_series_with_format():
    s1 = pd.Series([10, 20, 30])
    s2 = pd.Series([100, 200, 300])

    exp = pd.Series(['10.0%', '10.0%', '10.0%'])
    got = ratio(s1, s2, precision=2, percent=True, format=True)
    assert exp.equals(got)
