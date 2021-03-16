import numpy as np
import pandas as pd
import logging
from datetime import datetime
from datetime import date
from time import strptime

from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Pattern,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# apply_date {{{1
def apply_date(df: pd.DataFrame,
               columns: Union[str, List[str]],
               function: Callable, *args, **kwargs) -> pd.DataFrame:
    ''' Apply date function to multiple dataframe columns

    Usage example
    -------------
    %%piper
    sample_data()
    >> apply_date(['dates', 'order_dates'], to_julian)
    >> head()

    |   dates |   order_dates | countries   | regions   | ids   |   values_1 |   values_2 |
    |--------:|--------------:|:------------|:----------|:------|-----------:|-----------:|
    |  120001 |        120007 | Italy       | East      | A     |        311 |         26 |
    |  120002 |        120008 | Portugal    | South     | D     |        150 |        375 |
    |  120003 |        120009 | Spain       | East      | A     |        396 |         88 |
    |  120004 |        120010 | Italy       | East      | B     |        319 |        233 |


    %%piper
    sample_data()
    >> apply_date(['dates', 'order_dates'], fiscal_year, year_only=True)
    >> head()

    | dates    | order_dates   | countries   | regions   | ids   |   values_1 |   values_2 |
    |:---------|:--------------|:------------|:----------|:------|-----------:|-----------:|
    | FY 19/20 | FY 19/20      | Italy       | East      | A     |        311 |         26 |
    | FY 19/20 | FY 19/20      | Portugal    | South     | D     |        150 |        375 |
    | FY 19/20 | FY 19/20      | Spain       | East      | A     |        396 |         88 |
    | FY 19/20 | FY 19/20      | Italy       | East      | B     |        319 |        233 |


    Parameters
    ----------
    df : pandas dataframe

    columns : column(s) to apply function

    function : function to be applied.


    Return
    ------
    A pandas dataframe
    '''
    if columns is None:
        raise ValueError('Please specify function to apply')

    if isinstance(df, pd.Series):
        raise TypeError('Please specify DataFrame object')

    if function is None:
        raise ValueError('Please specify function to apply')

    if isinstance(columns, str):
        if columns not in df.columns:
            raise ValueError(f'column {columns} not found')

        df[columns] = df[columns].apply(function, *args, **kwargs)

    if isinstance(columns, list):
        for col in columns:
            if col not in df.columns:
                raise ValueError(f'column {col} not found')

            df[col] = df[col].apply(function, *args, **kwargs)

    return df


# from_julian {{{1
def from_julian(julian: Union[str, int],
                jde_format: bool = True) -> Any:
    ''' apply_date function: (JDE) Julian -> Gregorian

    References
    ----------
    # https://docs.oracle.com/cd/E26228_01/doc.93/e21961/julian_date_conv.htm#WEAWX259
    # http://nimishprabhu.com/the-mystery-of-jde-julian-date-format-solved.html

    Usage example
    -------------
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

    Parameters
    ----------
    julian date: julian date to be converted

    jde_format: default True. If False, assume 'standard' julian format.


    Returns
    -------
    gregorian_date: Any Gregorian based format
    '''
    if julian is None:
        return julian

    if isinstance(julian, date):
        return julian

    if isinstance(julian, int):
        julian = str(julian)

    try:
        if isinstance(julian, float):
            julian = str(int(julian))
    except ValueError:
        return julian

    if jde_format:
        if len(julian) != 6:
            return julian

        century = int(julian[0]) + 19
        year = julian[1:3]
        days = julian[3:6]
        std_julian = f'{century}{year}{days}'
        greg_date = datetime.strptime(std_julian, '%Y%j').date()
    else:
        format_ = '%Y%j' if len(julian) == 7 else '%y%j'
        greg_date = datetime.strptime(julian, format_).date()

    return greg_date


# to_julian {{{1
def to_julian(greg_date: Union[str, int], format: str = None):
    ''' apply_date function: Gregorian -> (JDE) Julian

    References
    ----------
    # https://docs.oracle.com/cd/E26228_01/doc.93/e21961/julian_date_conv.htm#WEAWX259
    # http://nimishprabhu.com/the-mystery-of-jde-julian-date-format-solved.html

    Usage example
    -------------
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


    Parameters
    ----------
    greg_date : gregorian format date (string)


    Return
    ------
    JDE Julian formatted string
    '''
    if greg_date in (np.NAN, pd.NaT, None, ''):
        return greg_date

    # if its an integer, convert to string first.
    if isinstance(greg_date, int):
        greg_date = str(greg_date)

    if format is None:
        format = '%Y-%m-%d'

    # Convert to pandas datetime format, then to Julian
    try:
        greg_date = pd.to_datetime(greg_date, format=format)
    except ValueError as _:
        return greg_date

    if isinstance(greg_date, datetime):
        century_prefix = str(greg_date.year)[:2]
        century_prefix = str(int(century_prefix) - 19)
        return int(century_prefix+greg_date.strftime('%y%j'))


# from_excel {{{1
def from_excel(excel_date: Union[str, float, int, pd.Timestamp]) -> pd.Timestamp:
    ''' apply_date function: excel serial date -> pd.Timestamp

    Converts excel serial format to pandas Timestamp object

    Usage example
    -------------
    assert from_excel(pd.Timestamp('2014-01-01 08:00:00')) == pd.Timestamp('2014-01-01 08:00:00')
    assert from_excel('41640.3333') == pd.Timestamp('2014-01-01 08:00:00')
    assert from_excel(41640.3333) == pd.Timestamp('2014-01-01 08:00:00')
    assert from_excel(44001) == pd.Timestamp('2020-06-19 00:00:00')
    assert from_excel('44001') == pd.Timestamp('2020-06-19 00:00:00')
    assert from_excel(43141) == pd.Timestamp('2018-02-10 00:00:00')
    assert from_excel('43962') == pd.Timestamp('2020-05-11 00:00:00')
    assert from_excel('') == ''
    assert from_excel(0) == 0
    assert pd.isna(from_excel(np.nan)) == pd.isna(np.nan)
    assert pd.isna(from_excel(pd.NaT)) == pd.isna(pd.NaT)


    Parameters
    ----------
    excel_date - serial excel date value


    Returns
    -------
    A pandas Timestamp object
    '''
    if isinstance(excel_date, pd.Timestamp):
        return excel_date

    if isinstance(excel_date, str):

        if len(excel_date) == 0:
            return excel_date

        try:
            excel_date = float(excel_date)
        except ValueError as e:
            excel_date = int(excel_date)
            return excel_date

    if excel_date == 0:
        return excel_date

    start_date = pd.to_datetime('1899-12-30')
    offset = pd.Timedelta(excel_date, 'd')

    if isinstance(excel_date, float):
        excel_date = start_date + offset
        excel_date = excel_date.round('1min')
    else:
        excel_date = start_date + offset

    return excel_date


# fiscal_year {{{1
def fiscal_year(date: Union[pd.Timestamp],
                start_month: int = 7,
                year_only: bool = False) -> str:
    ''' apply_date function: Convert to fiscal year

    Used with pd.Series date objects to obtain Fiscal Year information.

    Usage example
    -------------
    assert fiscal_year(pd.Timestamp('2014-01-01')) == 'FY 2013/2014'
    assert fiscal_year(pd.to_datetime('2014-01-01')) == 'FY 2013/2014'

    assert fiscal_year(pd.Timestamp('2014-01-01'), year_only=True) == 'FY 13/14'
    assert fiscal_year(pd.to_datetime('2014-01-01'), year_only=True) == 'FY 13/14'

    assert pd.isna(from_excel(np.nan)) == pd.isna(np.nan)
    assert pd.isna(from_excel(pd.NaT)) == pd.isna(pd.NaT)

    -----------
    df = pd.DataFrame()
    df['Date'] = pd.date_range('2020-01-01', periods=12, freq='M')
    df['Date'] = df['Date'].apply(fiscal_year)
    df.head()

    0     FY 2018/2019
    1     FY 2018/2019
    2     FY 2018/2019
    3     FY 2018/2019
    4     FY 2018/2019
    Name: Date, dtype: object

    Parameters
    ----------
    date : pd.TimeStamp object

    start_month : Fiscal year starting month, default 7 (Australia)

    year_only : False (default) - 4 digit year e.g. 2019, 2020
                True            - 2 digit year e.g. 19, 20
    '''
    prefix = 'FY'

    if year_only:
        year = int(str(date.year)[2:])
    else:
        year = date.year

    if date.month < start_month:
        text = f'{prefix} {year-1}/{year}'
    else:
        text = f'{prefix} {year}/{year+1}'

    return text
