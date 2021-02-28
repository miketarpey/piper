import pandas as pd
import logging
from datetime import datetime
from datetime import date
from time import strptime


logger = logging.getLogger(__name__)


# to_date {{{1
def to_date(df, columns=None, format='%d/%m/%Y',
            errors='ignore', function=None):
    ''' Given dataframe and list of columns, convert
    date string formatted data to pandas date data type.

    There are three 'function options:
    - using default pd.to_datetime() function
    - using built-in from_julian() and to_julian() functions
    - pass an external custom function to apply to the column data


    Example
    -------
    inventory = to_date(inventory, ['fifo', 'fefo'])

    julian_dt = to_date(df_dates, ['test_dates'], function='from_julian')

    dates = to_date(df_dates, ['test_dates'], function='to_julian',
                    format='%d/%m/%Y')

    convert_date = lambda x: pd.to_datetime(x, dayfirst=True,
                                            format='%d%m%Y', errors='coerce')

    actual = apply_date_function(df, columns=['test_dates'],
                                 function=convert_date)

    Example using %%piper
    ---------------------
    df <- pd.read_csv('inputs/export.dsv', sep='\t')
    >> clean_columns() >> trim()
    >> to_date(['adeftj', 'adexdj', 'adupmj'], function=from_julian)


    Parameters
    ----------
    df : pandas dataframe

    columns : list of columns names to apply function to

    format : default '%d/%m/%Y'
        date format string used to interpret column date data
        Assumption: Dates are expected in dd/mm/yyyy format

    errors : {‘ignore’, ‘raise’, ‘coerce’}, default ‘ignore’
        If ‘raise’, then invalid parsing will raise an exception
        If ‘coerce’, then invalid parsing will be set as NaT
        If ‘ignore’, then invalid parsing will return the input

    function : None -> Use default pd.to_datetime()
               'from_julian' -> convert using from_julian()
               'to_julian' -> convert using to_julian()

               function -> apply using user defined function passed

    Return
    ------
    pandas dataframe
    '''
    def convert_date(dt_value):

        return_dt = pd.to_datetime(dt_value, format=format, errors=errors)
        if isinstance(return_dt, pd.Timestamp):
            return_dt = return_dt.date()

        return return_dt

    if columns is None:
        raise ValueError('Error: No column information provided')

    if function is None:
        df = apply_date_function(df, columns, convert_date)
    elif function == 'from_julian':
        df = apply_date_function(df, columns, from_julian)
    elif function == 'to_julian':
        df = apply_date_function(df, columns, to_julian, format=format)
    else:
        df = apply_date_function(df, columns, function)

    return df


# apply_date_function {{{1
def apply_date_function(df, columns=None, function=None, format=None):
    ''' Apply date function to a number of dataframe columns

    Example
    -------
    df = apply_date_function(df, columns, convert_date)

    Parameters
    ----------
    df : pandas dataframe

    columns : list of columns names to apply function to

    function : function (reference) to be applied

    Return
    ------
    pandas dataframe

    '''
    if columns is None:
        raise ValueError('Error: No column information provided')

    if function is None:
        raise TypeError('Error: Invalid function reference provided')

    if isinstance(columns, str):

        if columns not in df.columns.values:
            raise TypeError(f'column {columns} not in dataframe.')

        func_name = function.__name__
        logger.debug(f'applying {func_name} to column {columns}')
        if format is None:
            df[columns] = df[columns].apply(function)
        else:
            df[columns] = df[columns].apply(function, format=format)


    if isinstance(columns, list):
        for column in columns:

            if column not in df.columns.values:
                raise TypeError(f'column {column} not in dataframe.')

            if column in df.columns.values:
                func_name = function.__name__
                logger.debug(f'applying {func_name} to column {column}')
                if format is None:
                    df[column] = df[column].apply(function)
                else:
                    df[column] = df[column].apply(function, format=format)

    return df


# from_julian {{{1
def from_julian(julian):
    ''' Convert jde julian format date (str or int) to datetime type

    References
    ----------
    # https://docs.oracle.com/cd/E26228_01/doc.93/e21961/julian_date_conv.htm#WEAWX259
    # http://nimishprabhu.com/the-mystery-of-jde-julian-date-format-solved.html

    Example
    -------
    from_julian('264144') -> datetime.date(2164, 5, 23)
    from_julian(264144)   -> datetime.date(2164, 5, 23)
    from_julian('117001').strftime('%b %#d, %Y') -> 'Jan 1, 2017'
    from_julian('098185').strftime('%b %#d, %Y') -> 'Jul 4, 1998'

    df[col] = df[col].apply(from_julian)

    answer = to_julian(from_julian(113338).strftime('%Y%m%d'))
    display(answer)
    113338

    answer = from_julian(to_julian(str(20120104))).strftime('%B,%_d %Y')
    display(answer)
    'January, 4 2012'

    Parameters
    ----------
    julian : JDE Julian formatted string (6 characters)

    Return
    ------
    datetime object
    '''
    if isinstance(julian, date):
        return julian

    if isinstance(julian, int):
        julian = str(julian)

    try:
        if isinstance(julian, float):
            julian = str(int(julian))
    except ValueError:
        # e.g. if passed value is NaN
        return julian

    if julian is None:
        return julian

    if len(julian) < 5 or len(julian) > 6:
        return julian
    else:
        if len(julian) == 5:
            julian = '0'+julian

        century = int(julian[0]) + 19
        year = julian[1:3]
        days = julian[3:6]

        greg_date = strptime(str(century)+year+days, '%Y%j')
        greg_date = datetime(*greg_date[:6]).date()

        return greg_date


# to_julian {{{1
def to_julian(*args, **kwargs):
    # def to_julian(greg_date, format='%Y-%m-%d', *args, **kwargs):
    ''' Convert gregorian date to jde julian format.

    References
    ----------
    # https://docs.oracle.com/cd/E26228_01/doc.93/e21961/julian_date_conv.htm#WEAWX259
    # http://nimishprabhu.com/the-mystery-of-jde-julian-date-format-solved.html

    Example
    -------
    today = datetime.now()
    julian_fmt = to_julian(today)

    greg_str = '30/7/2019'
    julian_fmt = to_julian(pd.to_datetime(greg_str))
    Result: 119211

    greg_str = '30/7/2019'
    julian_fmt = to_julian(greg_str, format='%d/%m/%Y')
    Result: 119211

    answer = to_julian(from_julian(113338).strftime('%Y%m%d'))
    display(answer)
    113338

    answer = from_julian(to_julian(str(20120104))).strftime('%B,%_d %Y')
    display(answer)
    'January, 4 2012'

    Parameters
    ----------
    greg_date : gregorian format date (string)

    Return
    ------
    JDE Julian formatted string
    '''
    greg_date = args[0]

    format = kwargs.get('format')
    if format is None:
        format = '%Y-%m-%d'

    if greg_date is None or greg_date == '':
        return greg_date

    try:
        greg_date = pd.to_datetime(greg_date, format=format)
    except ValueError as _:
        return greg_date

    if isinstance(greg_date, date):
        century_prefix = str(greg_date.year)[:2]
        century_prefix = str(int(century_prefix) - 19)
        return int(century_prefix+greg_date.strftime('%y%j'))


# from_excel_date {{{1
def from_excel_date(excel_date):
    '''
    converts excel float format to pandas datetime object
    '''
    if isinstance(excel_date, pd.Timestamp):
        return excel_date

    try:
        excel_date = int(excel_date)
    except ValueError as e:
        return excel_date

    start_date = pd.to_datetime('1899-12-30')
    offset = pd.DateOffset(excel_date, 'D')

    excel_date = start_date + offset

    return excel_date


# fiscal_year {{{1
def fiscal_year(date=None,
                start_month=7,
                year_only=False):
    '''
    Used with pd.Series date objects to obtain Fiscal Year information.

    Example
    -------
    df = pd.DataFrame()
    df['Date'] = pd.date_range('2020-01-01', periods=12, freq='M')
    df['Date'] = df['Date'].apply(fiscal_year)
    df

    0     FY 2018/2019
    1     FY 2018/2019
    2     FY 2018/2019
    3     FY 2018/2019
    4     FY 2018/2019
    5     FY 2018/2019
    6     FY 2019/2020
    7     FY 2019/2020
    8     FY 2019/2020
    9     FY 2019/2020
    10    FY 2019/2020
    11    FY 2019/2020
    12    FY 2019/2020
    13    FY 2019/2020
    14    FY 2019/2020
    15    FY 2019/2020
    16    FY 2019/2020
    17    FY 2019/2020
    18    FY 2020/2021
    19    FY 2020/2021
    20    FY 2020/2021
    21    FY 2020/2021
    22    FY 2020/2021
    23    FY 2020/2021
    Name: Date, dtype: object

    Parameters
    ----------
    date : pd.DateTimeStamp object

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
