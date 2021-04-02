from piper.custom import ratio
import datetime
import numpy as np
import pandas as pd
import pytest
from time import strptime
from piper.custom import add_xl_formula
from piper.factory import sample_data
from piper.factory import generate_periods, make_null_dates
from piper.custom import from_julian
from piper.custom import fiscal_year
from piper.custom import from_excel
from piper.custom import to_julian
from piper.verbs import across


# t_sample_data {{{1
@pytest.fixture
def t_sample_data():
    return sample_data()


# test_add_xl_formula {{{1
def test_add_xl_formula(t_sample_data):

    df = t_sample_data

    formula = '=CONCATENATE(A{row}, B{row}, C{row})'
    add_xl_formula(df, column_name='X7', formula=formula)

    expected = (367, )

    assert expected == df.X7.shape


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
def test_across_raise_column_parm_none():

    convert_date = lambda x: pd.to_datetime(x, dayfirst=True, format='%d%m%Y', errors='coerce')

    test = [30112019, 2942019, 3022019, 2822019, 2019430]
    got = pd.DataFrame(test, columns=['dates'])

    # Convert expected values to datetime format
    exp = ['30/11/2019', '29/4/2019', pd.NaT, '28/2/2019', pd.NaT]
    exp = pd.DataFrame(exp, columns=['dates'])
    exp.dates = exp.dates.astype('datetime64[ns]')

    got = across(got, columns=None, function=convert_date)

    assert exp.equals(got) == True


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

    exp = '100.0%'
    got = ratio(2.0, 2.0, percent=True)

    assert exp == got


# test_dividing_numbers_float_percent_with_round {{{1
def test_dividing_numbers_float_percent_with_round():
    ''' '''

    exp = 100.0000
    got = ratio(2.0, 2.0, percent=True, format=False, precision=4)

    assert exp == got

    exp = 50.00
    got = ratio(1.0, 2.0, percent=True, format=False, precision=2)

    assert exp == got


# test_dividing_numbers_int_percent_with_round {{{1
def test_dividing_numbers_int_percent_with_round():
    ''' '''

    exp = 100.0000
    got = ratio(2, 2, percent=True, format=False, precision=4)

    assert exp == got

    exp = 50.00
    got = ratio(1, 2, percent=True, format=False, precision=2)

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

# test_fiscal_year {{{1
def test_fiscal_year():

    assert fiscal_year(pd.Timestamp('2014-01-01')) == 'FY 2013/2014'
    assert fiscal_year(pd.to_datetime('2014-01-01')) == 'FY 2013/2014'

    assert fiscal_year(pd.Timestamp('2014-01-01'), year_only=True) == 'FY 13/14'
    assert fiscal_year(pd.to_datetime('2014-01-01'), year_only=True) == 'FY 13/14'

    assert pd.isna(from_excel(np.nan)) == pd.isna(np.nan)
    assert pd.isna(from_excel(pd.NaT)) == pd.isna(pd.NaT)
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



# test_from_julian_jde_format {{{1
def test_from_julian_jde_format():

    assert from_julian('093121') == datetime.date(1993, 5, 1)
    assert from_julian('1234') == '1234'
    assert from_julian('e1vh105b') == 'e1vh105b'
    assert from_julian('264144') == datetime.date(2164, 5, 23)
    assert from_julian(264144) == datetime.date(2164, 5, 23)
    assert from_julian(100001) == datetime.date(2000, 1, 1)
    assert from_julian('117001') == datetime.date(2017, 1, 1)
    assert from_julian('97032') == datetime.date(1997, 2, 1)
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


