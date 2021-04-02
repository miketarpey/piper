from datetime import date
from datetime import datetime
from time import strptime
import logging
import numpy as np #type: ignore
import pandas as pd #type: ignore
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_timedelta64_dtype
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


logger = logging.getLogger(__name__)


# add_xl_formula() {{{1
def add_xl_formula(df: pd.DataFrame,
                   column_name: str = 'xl_calc',
                   formula: str = '=CONCATENATE(A{row}, B{row}, C{row})',
                   offset: int = 2) -> pd.DataFrame:

    '''add Excel (xl) formula column

    Parameters
    ----------
    df
        pandas dataframe
    column_name
        the column name to be associated with the column formula values, default
        'xl_calc'
    formula
        Excel formula to be applied. As an example:

        .. code-block::

            '=CONCATENATE(A{row}, B{row}, C{row})'

        where {row} is the defined replacement variable which will be replaced
        with actual individual row value.
    offset
        starting row value, default = 2 (resultant xl sheet includes headers)


    Examples
    --------

    .. code-block::

        formula = '=CONCATENATE(A{row}, B{row}, C{row})'
        add_xl_formula(df, column_name='X7', formula=formula)


    Returns
    -------
    pandas dataframe
    '''
    col_values = []
    for x in range(offset, df.shape[0] + offset):
        repl_str = re.sub('{ROW}', str(x), string=formula, flags=re.I)
        col_values.append(repl_str)

    df[column_name] = col_values

    return df


# duration {{{1
def duration(s1: pd.Series,
             s2: pd.Series = None,
             unit: Union[str, None] = None,
             round: Union[bool, int] = 2,
             freq: str = 'd') -> pd.Series:
    ''' calculate duration between two columns (series)

    Parameters
    ----------
    s1
        'from' datetime series
    s2
        'to' datetime series.
        Default None. If None, defaults to today.
    interval
        default None - returns timedelta in days
                'd' - days as an integer,
                'years' (based on 365.25 days per year),
                'months' (based on 30 day month)

        Other possible options are:
            - ‘W’, ‘D’, ‘T’, ‘S’, ‘L’, ‘U’, or ‘N’
            - ‘days’ or ‘day’
            - ‘hours’, ‘hour’, ‘hr’, or ‘h’
            - ‘minutes’, ‘minute’, ‘min’, or ‘m’
            - ‘seconds’, ‘second’, or ‘sec’
            - ‘milliseconds’, ‘millisecond’, ‘millis’, or ‘milli’
            - ‘microseconds’, ‘microsecond’, ‘micros’, or ‘micro’-
            - ‘nanoseconds’, ‘nanosecond’, ‘nanos’, ‘nano’, or ‘ns’.

        check out pandas
        `timedelta object <https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html>`_
        for details.
    round
        Default False. If duration result is an integer and this
        parameter contains a positive integer, the result is round to this
        decimal precision.
    freq
        Default is 'd'(days). If the duration result is a pd.Timedelta dtype,
        the value can be 'rounded' using this frequency parameter.

        Must be a fixed frequency like 'S' (second) not 'ME' (month end).
        For a list of valid values, check out
        `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_


    Returns
    -------
    series
        if unit is None - series is of data type timedelta64[ns]
        otherwise series of type int.

    Examples
    --------

    .. code-block::

        %%piper
        sample_data()
        >> select(['-countries', '-regions', '-ids', '-values_1', '-values_2'])
        >> assign(new_date_col=pd.to_datetime('2018-01-01'))
        >> assign(duration = lambda x: duration(x.new_date_col, x.order_dates, unit='months'))
        >> assign(duration_dates_age = lambda x: duration(x['dates']))
        >> head(tablefmt='plain')

            dates       rder_dates new_date_col duration duration_dates_age
         0  2020-01-01  2020-01-07   2018-01-01       25           452 days
         1  2020-01-02  2020-01-08   2018-01-01       25           451 days
         2  2020-01-03  2020-01-09   2018-01-01       25           450 days
         3  2020-01-04  2020-01-10   2018-01-01       25           449 days

    '''
    if s2 is None:
        s2 = datetime.today()

    if unit is None:
        result = s2 - s1
    elif unit == 'years':
        result = ((s2 - s1) / pd.Timedelta(365.25, 'd'))
    elif unit == 'months':
        result = ((s2 - s1) / pd.Timedelta(30, 'd'))
    else:
        result = ((s2 - s1)) / pd.Timedelta(1, unit)

    if is_numeric_dtype(result):
        result = result.round(round)
    elif is_timedelta64_dtype(result):
        result = result.dt.round(freq=freq)

    return result


# factorize {{{1
def factorize(series: pd.Series,
              categories: List = None,
              ordered: int = False) -> pd.Series:
    ''' factorize / make column a categorical dtype

    Parameters
    ----------
    series
        pd.Series object to be converted to categorical
    categories
        list of unique category values within pd.Series
    ordered
        If true, categorical is ordered.


    Returns
    -------
    pd.Series
        Returned series with categorical data type

    Examples
    --------

    .. code-block::

        cat_order = ['Tops & Blouses', 'Beachwear',
                     'Footwear', 'Jeans', 'Sportswear']

        %%piper
        sample_sales()
        >> assign(product=lambda x: factorize(x['product'],
                                              categories=cat_order,
                                              ordered=True))
        >> group_by(['location', 'product'])
        >> summarise(Total=('actual_sales', 'sum'))
        >> unstack()
        >> flatten_cols(remove_prefix='Total')
        >> head(tablefmt='plain')

        location Tops & Blouses Beachwear  Footwear   Jeans Sportswear
        London           339236    388762    274674  404440     291561
        Milan            523052    368373    444624  364343     319199
        Paris            481787    464725    383093  178117     150222

    '''
    if categories is None:
        series = series.astype('category')
    else:
        category_dtype = pd.CategoricalDtype(categories=categories,
                                             ordered=ordered)
        series = series.astype(category_dtype)

    return series


# fiscal_year {{{1
def fiscal_year(date: Union[pd.Timestamp],
                start_month: int = 7,
                year_only: bool = False) -> str:
    '''Convert to fiscal year

    Used with pd.Series date objects to obtain Fiscal Year information.

    Parameters
    ----------
    date
        pd.TimeStamp object
    start_month
        Fiscal year starting month, default 7 (Australia)
    year_only
        Default False. Returns a 4 digit year e.g. 2019, 2020
                True   Returns a 2 digit year e.g. 19, 20

    Examples
    --------

    .. code-block::

        assert fiscal_year(pd.Timestamp('2014-01-01')) == 'FY 2013/2014'
        assert fiscal_year(pd.to_datetime('2014-01-01')) == 'FY 2013/2014'

        assert fiscal_year(pd.Timestamp('2014-01-01'), year_only=True) == 'FY 13/14'
        assert fiscal_year(pd.to_datetime('2014-01-01'), year_only=True) == 'FY 13/14'

        assert pd.isna(from_excel(np.nan)) == pd.isna(np.nan)
        assert pd.isna(from_excel(pd.NaT)) == pd.isna(pd.NaT)

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

    Parameters
    ----------
    julian date
        julian date to be converted

    jde_format
        default True. If False, assume 'standard' julian format.

    Returns
    -------
    gregorian_date: Any Gregorian based format

    Examples
    --------

    .. code-block::

        %%piper
        sample_sales()
        >> select(['-target_profit', '-actual_profit'])
        # >> assign(month = lambda x: x['month'].apply(to_julian))
        >> across('month', to_julian)
        >> head(5)

          location  product     month  target_sales  actual_sales
          London    Beachwear  121001         31749       29209.1
          London    Beachwear  121001         37833       34049.7
          London    Jeans      121001         29485       31549
          London    Jeans      121001         37524       40901.2
          London    Sportswear 121001         27216       29121.1
    '''
    if julian is None:
        return julian

    if isinstance(julian, str):
        if len(julian) > 6:
            return julian

    if isinstance(julian, date):
        return julian

    if isinstance(julian, int):

        if jde_format:
            julian = str(julian).zfill(6)
        else:
            julian = str(julian)

    try:
        if isinstance(julian, float):
            julian = str(int(julian))
    except ValueError:
        return julian

    if jde_format:

        if int(julian) < 70001:
            return julian

        if len(julian) > 6 or len(julian) < 5:
            return julian

        if isinstance(julian, str):
            julian = julian.zfill(6)

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
def from_excel(excel_date: Union[str, float, int, pd.Timestamp]
               ) -> pd.Timestamp:
    ''' apply_date function: excel serial date -> pd.Timestamp

    Converts excel serial format to pandas Timestamp object

    Examples
    --------

    .. code-block::

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
        except ValueError as _:
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
          format: bool = True) -> Any:
    ''' Calculate the Ratio / percentage of two values

    Custom function which calculate the ratio and optionally
    percentage of two values or series.

    .. note::

        Passes back np.inf value for 'divide by zero' use case.

    Parameters
    ----------
    value1
        integer, float, or pd.Series
    value2
        integer, float, or pd.Series
    precision
        Default 2. Returned result is rounded to precision value.
    percent
        Default False. If True, calculates the percentage.
    format
        Default False. If True, returns a string, formatted as a
        percentage value e.g. 92.0%


    Returns
    -------
    float - if values1 and 2 are single int/float values
    pd.Series - if values1 and 2 are pd.Series

    Examples
    --------
    .. code-block::

        s1 = pd.Series([10, 20, 30])
        s2 = pd.Series([1.3, 5.4, 3])
        ratio(s1, s2)

                   0
           0    7.69
           1    3.70

    .. code-block::

        %%piper

        sample_sales()
        >> select(['-target_profit', '-actual_profit'])
        >> assign(std_ratio = lambda x: x.actual_sales / x.target_sales)
        >> assign(ratio = lambda x: ratio(x.actual_sales, x.target_sales,
                                          percent=True, format=True, precision=4))
        >> head(10)

        location  product     month      target_sales  actual_sales  std_ratio   ratio
        London    Beachwear   2021-01-01        31749       29209.1       0.92   92.0%
        London    Beachwear   2021-01-01        37833       34049.7       0.9    90.0%
        London    Jeans       2021-01-01        29485       31549         1.07   107.0%
        London    Jeans       2021-01-01        37524       40901.2       1.09   109.0%
        London    Sportswear  2021-01-01        27216       29121.1       1.07   107.0%



    '''
    if isinstance(value1, pd.Series):

        if percent:
            result = (value1 * 100 / value2)

            if precision is not None:
                result = result.round(precision)

            if format:
                result = result.astype(str) + '%'
        else:
            result = (value1 / value2)

            if precision is not None:
                result = result.round(precision)

        return result

    # Assumption, if not pd.Series, we are dealing
    # with individual int or float values
    try:
        if percent:
            result = ((value1 * 100) / value2)

            if precision is not None:
                result = round(result, precision)

            if format:
                result = f'{result}%'
        else:
            result = value1 / value2

            if precision is not None:
                result = round(result, precision)

    except ZeroDivisionError as e:
        logger.info(e)
        return np.inf

    return result


# to_julian {{{1
def to_julian(greg_date: Union[str, int], format: str = None):
    ''' apply_date function: Gregorian -> (JDE) Julian

    Parameters
    ----------
    greg_date : gregorian format date (string)


    Returns
    -------
    JDE Julian formatted string


    References
    ----------
    # https://docs.oracle.com/cd/E26228_01/doc.93/e21961/julian_date_conv.htm#WEAWX259
    # http://nimishprabhu.com/the-mystery-of-jde-julian-date-format-solved.html

    Examples
    --------

    .. code-block::

        %%piper

        sample_sales()
        >> select(['-target_profit', '-actual_profit'])
        # >> assign(month = lambda x: x['month'].apply(to_julian))
        >> across('month', to_julian)
        >> assign(month = lambda x: x['month'].apply(from_julian))
        # >> across('month', from_julian)
        >> head(5)

          location  product     month     target_sales  actual_sales
          London    Beachwear  2021-01-01        31749       29209.1
          London    Beachwear  2021-01-01        37833       34049.7
          London    Jeans      2021-01-01        29485       31549
          London    Jeans      2021-01-01        37524       40901.2
          London    Sportswear 2021-01-01        27216       29121.1
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
