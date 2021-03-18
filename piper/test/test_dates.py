import pytest
import pandas as pd
import numpy as np
import datetime
from time import strptime
from piper.factory import generate_periods, make_null_dates
from piper.dates import from_julian
from piper.dates import fiscal_year
from piper.dates import from_excel
from piper.dates import to_julian
from piper.verbs import across


# test_across_str_date_single_col_pd_to_datetime {{{1
def test_across_str_date_single_col_pd_to_datetime():
    ''' '''
    test = ['30/11/2019', '29/4/2019', '30/2/2019', '28/2/2019', '2019/4/30']
    got = pd.DataFrame(test, columns=['dates'])

    # Convert expected values to datetime format
    exp = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    exp = pd.DataFrame(exp, columns=['dates'])
    exp.dates = exp.dates.astype('datetime64[ns]')

    got = across(got, 'dates', pd.to_datetime, format='%d/%m/%Y', errors='coerce')

    assert exp.equals(got) == True


# test_across_str_date_single_col_lambda {{{1
def test_across_str_date_single_col_lambda():
    ''' '''
    convert_date = lambda x: pd.to_datetime(x, dayfirst=True, format='%d%m%Y', errors='coerce')

    test = [30112019, 2942019, 3022019, 2822019, 2019430]
    got = pd.DataFrame(test, columns=['dates'])

    # Convert expected values to datetime format
    exp = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    exp = pd.DataFrame(exp, columns=['dates'])
    exp.dates = exp.dates.astype('datetime64[ns]')

    got = across(got, 'dates', convert_date)

    assert exp.equals(got) == True


# test_across_raise_column_parm_none_ValueError {{{1
def test_across_raise_column_parm_none_ValueError():

    convert_date = lambda x: pd.to_datetime(x, dayfirst=True, format='%d%m%Y', errors='coerce')

    test = [30112019, 2942019, 3022019, 2822019, 2019430]
    got = pd.DataFrame(test, columns=['dates'])

    # Convert expected values to datetime format
    exp = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    exp = pd.DataFrame(exp, columns=['dates'])
    exp.dates = exp.dates.astype('datetime64[ns]')

    with pytest.raises(ValueError):
        got = across(got, columns=None, function=convert_date)


# test_across_raise_function_parm_none_ValueError {{{1
def test_across_raise_function_parm_none_ValueError():

    convert_date = lambda x: pd.to_datetime(x, dayfirst=True, format='%d%m%Y', errors='coerce')

    test = [30112019, 2942019, 3022019, 2822019, 2019430]
    got = pd.DataFrame(test, columns=['dates'])

    # Convert expected values to datetime format
    exp = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    exp = pd.DataFrame(exp, columns=['dates'])
    exp.dates = exp.dates.astype('datetime64[ns]')

    with pytest.raises(ValueError):
        got = across(got, columns='dates', function=None)


# test_across_raise_Series_parm_TypeError {{{1
def test_across_raise_Series_parm_TypeError():

    convert_date = lambda x: pd.to_datetime(x, dayfirst=True, format='%d%m%Y', errors='coerce')

    test = [30112019, 2942019, 3022019, 2822019, 2019430]
    got = pd.DataFrame(test, columns=['dates'])

    # Convert expected values to datetime format
    exp = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    exp = pd.DataFrame(exp, columns=['dates'])
    exp.dates = exp.dates.astype('datetime64[ns]')

    with pytest.raises(TypeError):
        got = across(pd.Series(test), columns='dates', function=convert_date)


# test_across_raise_column_parm_ValueError {{{1
def test_across_raise_column_parm_ValueError():

    convert_date = lambda x: pd.to_datetime(x, dayfirst=True, format='%d%m%Y', errors='coerce')

    test = [30112019, 2942019, 3022019, 2822019, 2019430]
    got = pd.DataFrame(test, columns=['dates'])

    # Convert expected values to datetime format
    exp = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    exp = pd.DataFrame(exp, columns=['dates'])
    exp.dates = exp.dates.astype('datetime64[ns]')

    with pytest.raises(ValueError):
        got = across(got, columns='invalid', function=convert_date)


# test_across_dataframe_single_column_with_lambda {{{1
def test_across_dataframe_single_column_with_lambda():

    convert_date = lambda x: x.strftime('%b %-d, %Y') if not x is pd.NaT else x

    df = generate_periods(delta_range=(1, 10), rows=20)
    df = make_null_dates(df, null_values_percent=.2)

    exp = df.copy(deep=True)
    exp.effective = exp.effective.apply(convert_date)

    got = across(df, columns='effective', function=convert_date)

    assert exp.equals(got) == True


# test_across_dataframe_multiple_columns_with_lambda {{{1
def test_across_dataframe_multiple_columns_with_lambda():

    convert_date = lambda x: x.strftime('%b %-d, %Y') if not x is pd.NaT else x

    df = generate_periods(delta_range=(1, 10), rows=20)
    df = make_null_dates(df, null_values_percent=.2)

    exp = df.copy(deep=True)
    exp.effective = exp.effective.apply(convert_date)
    exp.expired = exp.expired.apply(convert_date)

    got = across(df, columns=['effective', 'expired'], function=convert_date)

    assert exp.equals(got) == True


# test_across_dataframe_multiple_columns_raise_invalid_column {{{1
def test_across_dataframe_multiple_columns_raise_invalid_column():

    convert_date = lambda x: x.strftime('%b %-d, %Y') if not x is pd.NaT else x

    df = generate_periods(delta_range=(1, 10), rows=20)
    df = make_null_dates(df, null_values_percent=.2)

    exp = df.copy(deep=True)
    exp.effective = exp.effective.apply(convert_date)
    exp.expired = exp.expired.apply(convert_date)

    with pytest.raises(ValueError):
        got = across(df, columns=['effective', 'invalid'], function=convert_date)


# test_from_julian_jde_format {{{1
def test_from_julian_jde_format():

    assert from_julian('093121') == datetime.date(1993, 5, 1)
    assert from_julian('1234') == '1234'
    assert from_julian('264144') == datetime.date(2164, 5, 23)
    assert from_julian(264144) == datetime.date(2164, 5, 23)
    assert from_julian(100001) == datetime.date(2000, 1, 1)
    assert from_julian('117001') == datetime.date(2017, 1, 1)
    assert from_julian('17001', jde_format=False) == datetime.date(2017, 1, 1)
    assert from_julian(17001, jde_format=False) == datetime.date(2017, 1, 1)
    assert from_julian(100001.0) == datetime.date(2000, 1, 1)
    assert from_julian(None) == None
    assert pd.isna(from_julian(np.nan)) == pd.isna(np.nan)
    assert pd.isna(from_julian(pd.NaT)) == pd.isna(pd.NaT)


# test_to_julian_jde_format {{{1
def test_to_julian_jde_format():

    assert to_julian(pd.to_datetime('30/7/2019')) == 119211
    assert to_julian(30072019, format='%d%m%Y') == 119211
    assert to_julian('30/07/2019', format='%d/%m/%Y') == 119211
    assert to_julian('2019/07/30', format='%Y/%m/%d') == 119211
    assert to_julian('2019-07-30') == 119211
    assert to_julian('30072019', format='%d%m%Y') == 119211
    assert to_julian('--') == '--'
    assert to_julian('') == ''
    assert pd.isna(to_julian(np.nan)) == pd.isna(np.nan)
    assert pd.isna(to_julian(pd.NaT)) == pd.isna(pd.NaT)


# test_from_excel_date {{{1
def test_from_excel_date():

    assert from_excel(pd.Timestamp('2014-01-01 08:00:00')) == pd.Timestamp('2014-01-01 08:00:00')
    assert from_excel('41640.3333') == pd.Timestamp('2014-01-01 08:00:00')
    assert from_excel(41640.3333) == pd.Timestamp('2014-01-01 08:00:00')
    assert from_excel(44001) == pd.Timestamp('2020-06-19 00:00:00')
    assert from_excel('44001') == pd.Timestamp('2020-06-19 00:00:00')
    assert from_excel(43141) == pd.Timestamp('2018-02-10 00:00:00')
    assert from_excel('43962') == pd.Timestamp('2020-05-11 00:00:00')
    assert from_excel('43962') == pd.Timestamp('2020-05-11 00:00:00')
    assert from_excel('03/30/2013') == '03/30/2013'
    assert from_excel('') == ''
    assert from_excel(0) == 0
    assert pd.isna(from_excel(np.nan)) == pd.isna(np.nan)
    assert pd.isna(from_excel(pd.NaT)) == pd.isna(pd.NaT)



# test_fiscal_year {{{1
def test_fiscal_year():

    assert fiscal_year(pd.Timestamp('2014-01-01')) == 'FY 2013/2014'
    assert fiscal_year(pd.to_datetime('2014-01-01')) == 'FY 2013/2014'

    assert fiscal_year(pd.Timestamp('2014-01-01'), year_only=True) == 'FY 13/14'
    assert fiscal_year(pd.to_datetime('2014-01-01'), year_only=True) == 'FY 13/14'

    assert pd.isna(from_excel(np.nan)) == pd.isna(np.nan)
    assert pd.isna(from_excel(pd.NaT)) == pd.isna(pd.NaT)
