import pytest
import pandas as pd

from piper.styler import boolean_filter


# test_boolean_filter_series_max(): {{{1
def test_boolean_filter_series_max():
    """
    """
    data = {'0': 6.2900347819,
            '1': 6.1180979358,
            '2': 5.6076394095,
            '3': None,
            '4': 6.0367669286,
            '5': 5.7629990307,
            '6': 4.9308246632,
            '7': 6.9313042709,
            '8': 5.6662223272,
            '9': 6.0294623651}

    series = pd.Series(data)

    expected = [False, False, False, False,
                False, False, False, True,
                False, False]

    got = list(boolean_filter(series, operator='max'))

    assert expected == got


# test_boolean_filter_series_min(): {{{1
def test_boolean_filter_series_min():
    """
    """

    data = {'0': 6.2900347819,
            '1': 6.1180979358,
            '2': 5.6076394095,
            '3': None,
            '4': 6.0367669286,
            '5': 5.7629990307,
            '6': 4.9308246632,
            '7': 6.9313042709,
            '8': 5.6662223272,
            '9': 6.0294623651}

    series = pd.Series(data)

    expected = [False, False, False, False,
                False, False, True, False,
                False, False]

    got = list(boolean_filter(series, operator='min'))

    assert expected == got


# test_boolean_filter_series_null(): {{{1
def test_boolean_filter_series_null():
    """
    """

    data = {'0': 6.2900347819,
            '1': 6.1180979358,
            '2': 5.6076394095,
            '3': None,
            '4': 6.0367669286,
            '5': 5.7629990307,
            '6': 4.9308246632,
            '7': 6.9313042709,
            '8': 5.6662223272,
            '9': 6.0294623651}

    series = pd.Series(data)

    expected = [False, False, False, True,
                False, False, False, False,
                False, False]

    got = list(boolean_filter(series, operator='null'))

    assert expected == got


# test_boolean_filter_series_notnull(): {{{1
def test_boolean_filter_series_notnull():
    """
    """

    data = {'0': 6.2900347819,
            '1': 6.1180979358,
            '2': 5.6076394095,
            '3': None,
            '4': 6.0367669286,
            '5': 5.7629990307,
            '6': 4.9308246632,
            '7': 6.9313042709,
            '8': 5.6662223272,
            '9': 6.0294623651}

    series = pd.Series(data)

    expected = [True, True, True, False, True,
                True, True, True, True, True]

    got = list(boolean_filter(series, operator='!null'))

    assert expected == got


# test_boolean_filter_series_lt(): {{{1
def test_boolean_filter_series_lt():
    """
    """

    data = {'0': 6.2900347819,
            '1': 6.1180979358,
            '2': 5.6076394095,
            '3': None,
            '4': 6.0367669286,
            '5': 5.7629990307,
            '6': 4.9308246632,
            '7': 6.9313042709,
            '8': 5.6662223272,
            '9': 6.0294623651}

    series = pd.Series(data)

    expected = [False, False, False, False, False, False,
                True, False, False, False]

    got = list(boolean_filter(series, operator='<', value=5))

    assert expected == got


# test_boolean_filter_series_gt(): {{{1
def test_boolean_filter_series_gt():
    """
    """

    data = {'0': 6.2900347819,
            '1': 6.1180979358,
            '2': 5.6076394095,
            '3': None,
            '4': 6.0367669286,
            '5': 5.7629990307,
            '6': 4.9308246632,
            '7': 6.9313042709,
            '8': 5.6662223272,
            '9': 6.0294623651}

    series = pd.Series(data)

    expected = [True, True, True, False, True,
                True, False, True, True, True]

    got = list(boolean_filter(series, operator='>', value=5))

    assert expected == got


# test_boolean_filter_series_eq(): {{{1
def test_boolean_filter_series_eq():
    """
    """

    data = {'0': 6.2900347819,
            '1': 6.1180979358,
            '2': 5.6076394095,
            '3': None,
            '4': 6.0367669286,
            '5': 5.7629990307,
            '6': 4.9308246632,
            '7': 6.9313042709,
            '8': 5.6662223272,
            '9': 6.0294623651}

    series = pd.Series(data)

    expected = [False, False, True, False, False, False,
                False, False, False, False]

    got = list(boolean_filter(series, value=5.6076394095))

    assert expected == got


# test_boolean_filter_num_vs_num_gt(): {{{1
def test_boolean_filter_nume_vs_num_gt():
    """
    """
    expected = True

    got = boolean_filter(50, operator='>', value=40)

    assert expected == got


# test_boolean_filter_num_vs_num_equals(): {{{1
def test_boolean_filter_num_vs_num_equals():
    """
    """
    expected = False

    got = boolean_filter(50, operator='=', value=40)

    assert expected == got




# test_boolean_filter_string_equals(): {{{1
def test_boolean_filter_string_equals():
    """
    """
    expected = True

    got = boolean_filter('A', operator='=', value='A')

    assert expected == got


# test_boolean_filter_string_lt(): {{{1
def test_boolean_filter_string_lt():
    """
    """
    expected = True

    got = boolean_filter('A', operator='<', value='B')

    assert expected == got


# test_boolean_filter_string_gt(): {{{1
def test_boolean_filter_string_gt():
    """
    """
    expected = False

    got = boolean_filter('A', operator='>', value='B')

    assert expected == got
