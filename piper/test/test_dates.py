import pytest
import pandas as pd
import numpy as np
import datetime
from time import strptime
from piper.dates import to_date
from piper.dates import from_julian
from piper.dates import to_julian
from piper.dates import apply_date_function


def test_to_date_raise_column_parm_none_error():

    dates_list = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    expected = pd.Series(dates_list).apply(pd.to_datetime)
    expected = expected.astype('datetime64[ns]')

    dates_list = [30112019, 2942019, 3022019, 2822019, 2019430]
    df = pd.DataFrame(dates_list, columns=['test_dates'])

    convert_date = lambda x: pd.to_datetime(x, dayfirst=True,
                                            format='%d%m%Y', errors='coerce')

    # Set parameter columns = None (will raise error)
    with pytest.raises(Exception):
        to_date(df, columns=None, function=convert_date) is None


def test_str_to_date():

    dates_list = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    expected = pd.Series(dates_list).apply(pd.to_datetime)
    expected = expected.astype('datetime64[ns]')

    dates_list = ['30/11/2019', '29/4/2019', '30/2/2019',
                  '28/2/2019', '2019/4/30']
    df_dates = pd.DataFrame(dates_list, columns=['test_dates'])

    actual = to_date(df_dates, ['test_dates'], format='%d/%m/%Y',
                     errors='coerce')
    actual = pd.Series(actual.test_dates)
    actual = actual.astype('datetime64[ns]')

    assert expected[1] == actual[1]


def test_to_date_from_julian():

    dates_list = ['23/05/2164', '30/07/2019']
    expected = pd.Series(dates_list).apply(pd.to_datetime)
    expected = expected.astype('datetime64[ns]')

    julian_dates = [264144, 119211]
    df_dates = pd.DataFrame(julian_dates, columns=['test_dates'])

    actual = to_date(df_dates, ['test_dates'], function='from_julian')
    actual = pd.Series(actual.test_dates)
    actual = actual.astype('datetime64[ns]')

    assert expected[0] == actual[0]
    assert expected[1] == actual[1]


def test_to_date_to_julian():

    julian_dates = [264144, 119211]
    expected = pd.DataFrame(julian_dates, columns=['test_dates'])

    dates_list = ['23/05/2164', '30/07/2019']
    df_dates = pd.DataFrame(dates_list, columns=['test_dates'])

    actual = to_date(df_dates, ['test_dates'], function='to_julian',
                     format='%d/%m/%Y')

    assert expected.loc[0, 'test_dates'] == actual.loc[0, 'test_dates']
    assert expected.loc[1, 'test_dates'] == actual.loc[1, 'test_dates']


def test_to_date_apply_function():

    dates_list = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    expected = pd.Series(dates_list).apply(pd.to_datetime)
    expected = expected.astype('datetime64[ns]')

    dates_list = [30112019, 2942019, 3022019, 2822019, 2019430]
    df = pd.DataFrame(dates_list, columns=['test_dates'])

    convert_date = lambda x: pd.to_datetime(x, dayfirst=True,
                                            format='%d%m%Y', errors='coerce')

    actual = to_date(df, ['test_dates'], function=convert_date)

    actual = pd.Series(actual.test_dates)
    actual = actual.astype('datetime64[ns]')

    assert expected[1] == actual[1]


def test_apply_date():

    dates_list = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    expected = pd.Series(dates_list).apply(pd.to_datetime)
    expected = expected.astype('datetime64[ns]')

    dates_list = [30112019, 2942019, 3022019, 2822019, 2019430]
    df = pd.DataFrame(dates_list, columns=['test_dates'])

    convert_date = lambda x: pd.to_datetime(x, dayfirst=True,
                                            format='%d%m%Y', errors='coerce')

    # Set parameter columns != None (will convert specified columns)
    actual = apply_date_function(df, columns=['test_dates'],
                                 function=convert_date)

    actual = pd.Series(actual.test_dates)
    actual = actual.astype('datetime64[ns]')

    assert expected[1] == actual[1]


def test_apply_date_raise_column_parm_none_error():

    dates_list = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    expected = pd.Series(dates_list).apply(pd.to_datetime)
    expected = expected.astype('datetime64[ns]')

    dates_list = [30112019, 2942019, 3022019, 2822019, 2019430]
    df = pd.DataFrame(dates_list, columns=['test_dates'])

    convert_date = lambda x: pd.to_datetime(x, dayfirst=True,
                                            format='%d%m%Y', errors='coerce')

    # Set parameter columns = None (will raise error)
    with pytest.raises(Exception):
        apply_date_function(df, columns=None, function=convert_date) is None


def test_apply_date_raise_column_parm_invalid_colname_error():

    dates_list = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    expected = pd.Series(dates_list).apply(pd.to_datetime)
    expected = expected.astype('datetime64[ns]')

    dates_list = [30112019, 2942019, 3022019, 2822019, 2019430]
    df = pd.DataFrame(dates_list, columns=['test_dates'])

    convert_date = lambda x: pd.to_datetime(x, dayfirst=True,
                                            format='%d%m%Y', errors='coerce')

    # Set parameter columns = None (will raise error)
    with pytest.raises(Exception):
        apply_date_function(df, columns=['invalid'],
                            function=convert_date) is None


def test_apply_date_raise_function_parm_none_error():

    dates_list = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    expected = pd.Series(dates_list).apply(pd.to_datetime)
    expected = expected.astype('datetime64[ns]')

    dates_list = [30112019, 2942019, 3022019, 2822019, 2019430]
    df = pd.DataFrame(dates_list, columns=['test_dates'])

    # Set parameter function = None (will raise error)
    with pytest.raises(Exception):
        apply_date_function(df, columns=['test_dates'], function=None) is None


def test_int_to_date():

    dates_list = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    expected = pd.Series(dates_list).apply(pd.to_datetime)
    expected = expected.astype('datetime64[ns]')

    dates_list = [30112019, 2942019, 3022019, 2822019, 2019430]
    df_dates = pd.DataFrame(dates_list, columns=['test_dates'])

    actual = to_date(df_dates, ['test_dates'], format='%d%m%Y',
                     errors='coerce')
    actual = pd.Series(actual.test_dates)
    actual = actual.astype('datetime64[ns]')

    assert expected[1] == actual[1]


def test_from_julian_with_value_less_than_six_chars():

    expected = strptime('1/5/1993', '%d/%m/%Y')
    expected = datetime.datetime(*expected[:6]).date()
    actual = from_julian('93121')

    assert expected == actual


def test_from_julian_with_value_less_than_five_chars():

    expected = '1234'
    actual = from_julian('1234')

    assert expected == actual


def test_from_julian_with_string_value():

    expected = datetime.date(2164, 5, 23)
    actual = from_julian('264144')

    assert expected == actual


def test_from_julian_with_int_value():

    expected = datetime.date(2164, 5, 23)
    actual = from_julian(264144)

    assert expected == actual

    expected = datetime.date(2000, 1, 1)
    actual = from_julian(100001)

    assert expected == actual


def test_from_julian_with_NaT():

    expected = pd.NaT
    actual = from_julian(pd.NaT)

    assert pd.isnull(expected)
    assert pd.isnull(actual)


def test_from_julian_with_float_value():

    expected = datetime.date(2164, 5, 23)
    actual = from_julian(264144)

    assert expected == actual

    expected = datetime.date(2000, 1, 1)
    actual = from_julian(100001.0)

    assert expected == actual


def test_from_julian_with_NaN():

    # expected = np.nan
    actual = from_julian(np.nan)

    assert np.isnan(actual)


def test_from_julian_with_None():

    # expected = np.nan
    actual = from_julian(None)

    assert actual == None


def test_from_julian_with_special_str_fmt():

    expected = 'Jan 1, 2017'
    # actual = from_julian('117001').strftime('%b %#d, %Y')
    # linux version (leading zero)
    actual = from_julian('117001').strftime('%b %-d, %Y')

    assert expected == actual


def test_to_julian_with_special_str_fmt():

    expected = 119211
    greg_str = '30/7/2019'
    actual = to_julian(pd.to_datetime(greg_str))

    assert expected == actual

    greg_str = '30/7/2019'
    actual = to_julian(greg_str, format='%d/%m/%Y')

    assert expected == actual


def test_to_julian_with_int_value():

    expected = 119211

    greg_str = 30072019
    actual = to_julian(greg_str, format='%d%m%Y')

    assert expected == actual


def test_to_julian_with_strange_value():

    expected = '--'
    greg_str = '--'
    actual = to_julian(greg_str, format='%d%m%Y')

    assert expected == actual

    expected = ''
    greg_str = ''
    actual = to_julian(greg_str, format='%d%m%Y')

    assert expected == actual


def test_to_julian_with_ValueError2():

    greg_dt = None
    greg_dt = ''

    with pytest.raises(Exception):
        assert to_julian(greg_dt)
