from datetime import date
from datetime import datetime
from time import strptime
import logging
import logging
import numpy as np #type: ignore
import pandas as pd #type: ignore
import re
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    %%piper

    sample_sales()
    >> select(['-target_profit', '-actual_profit'])
    # >> assign(month = lambda x: x['month'].apply(to_julian))
    >> across('month', to_julian)
    >> head(5)

    | location| product   | month |target_sales |actual_sales |
    |:--------|:----------|------:|------------:|------------:|
    | London  | Beachwear |121001 |       31749 |     29209.1 |
    | London  | Beachwear |121001 |       37833 |     34049.7 |
    | London  | Jeans     |121001 |       29485 |     31549   |
    | London  | Jeans     |121001 |       37524 |     40901.2 |
    | London  | Sportswear|121001 |       27216 |     29121.1 |

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

        if re.search(r'[\\\/\-]', excel_date):
            return excel_date

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


# ratio {{{1
def ratio(value1: Union[int, float, pd.Series],
          value2: Union[int, float, pd.Series],
          precision: int = 2,
          percent: bool = False,
          format: bool = False) -> Any:
    ''' ratio - ratio/percentage of two values (or pd.Series)

    Provide parameters to adjust decimal rounding and a 'format'
    option to send back string rather than numeric with % appended.
    e.g. '33.43%'

    Passes back np.inf value for 'divide by zero' use case.

    Usage example
    -------------
    s1 = pd.Series([10, 20, 30])
    s2 = pd.Series([1.3, 5.4, 3])
    ratio(s1, s2)

    |    |     0 |
    |---:|------:|
    |  0 |  7.69 |
    |  1 |  3.70 |

    %%piper

    sample_sales()
    >> select(['-target_profit', '-actual_profit'])
    >> assign(std_ratio = lambda x: x.actual_sales / x.target_sales)
    >> assign(ratio = lambda x: ratio(x.actual_sales, x.target_sales,
                                      percent=True, format=True, precision=4))
    >> head(10)

    | location| product   | month     |target_sales |actual_sales |std_ratio | ratio  |
    |:--------|:----------|:----------|------------:|------------:|---------:|:-------|
    | London  | Beachwear | 2021-01-01|       31749 |     29209.1 |     0.92 | 92.0%  |
    | London  | Beachwear | 2021-01-01|       37833 |     34049.7 |     0.9  | 90.0%  |
    | London  | Jeans     | 2021-01-01|       29485 |     31549   |     1.07 | 107.0% |
    | London  | Jeans     | 2021-01-01|       37524 |     40901.2 |     1.09 | 109.0% |
    | London  | Sportswear| 2021-01-01|       27216 |     29121.1 |     1.07 | 107.0% |


    Parameters
    ----------
    value1: integer, float, or pd.Series

    value2: integer, float, or pd.Series


    Returns
    -------
    float - if values1 and 2 are single int/float values
    pd.Series - if values1 and 2 are pd.Series

    '''
    if isinstance(value1, pd.Series):
        if percent:
            result = (value1 * 100 / value2)
        else:
            result = (value1 / value2)

        if precision is not None:
            result = result.round(precision)

        if format:
            result = result.astype(str) + '%'

        return result

    # Assumption, if not pd.Series, we are dealing
    # with individual int or float values
    try:
        if percent:
            result = ((value1 * 100)/ value2)
        else:
            result = value1 / value2

        if precision is not None:
            result = round(result, precision)

        if format:
            result = f'{result}%'

    except ZeroDivisionError as e:
        logger.info(e)
        return np.inf

    return result


# to_julian {{{1
def to_julian(greg_date: Union[str, int], format: str = None):
    ''' apply_date function: Gregorian -> (JDE) Julian

    References
    ----------
    # https://docs.oracle.com/cd/E26228_01/doc.93/e21961/julian_date_conv.htm#WEAWX259
    # http://nimishprabhu.com/the-mystery-of-jde-julian-date-format-solved.html

    Usage example
    -------------
    %%piper

    sample_sales()
    >> select(['-target_profit', '-actual_profit'])
    # >> assign(month = lambda x: x['month'].apply(to_julian))
    >> across('month', to_julian)
    >> assign(month = lambda x: x['month'].apply(from_julian))
    # >> across('month', from_julian)
    >> head(5)

    | location| product   | month    |target_sales |actual_sales |
    |:--------|:----------|---------:|------------:|------------:|
    | London  | Beachwear |2021-01-01|       31749 |     29209.1 |
    | London  | Beachwear |2021-01-01|       37833 |     34049.7 |
    | London  | Jeans     |2021-01-01|       29485 |     31549   |
    | London  | Jeans     |2021-01-01|       37524 |     40901.2 |
    | London  | Sportswear|2021-01-01|       27216 |     29121.1 |


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


