import pandas as pd
import numpy as np
from xlsxwriter.utility import xl_col_to_name
import logging
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


# get_sample_data {{{1
def get_sample_data(start: int = 2020,
                    end: int = 2021,
                    freq: str = 'D',
                    seed: int = 30) -> pd.DataFrame:
    ''' TEST - Regions, countries, dates, values

    Usage example
    -------------
    get_sample_data().head()

    | dates      | order_dates| countries| regions| ids|values_1 |values_2 |
    |:-----------|:-----------|:---------|:-------|:---|--------:|--------:|
    | 2020-01-01 | 2020-01-07 | Italy    | East   | A  |     311 |      26 |
    | 2020-01-02 | 2020-01-08 | Portugal | South  | D  |     150 |     375 |
    | 2020-01-03 | 2020-01-09 | Spain    | East   | A  |     396 |      88 |
    | 2020-01-04 | 2020-01-10 | Italy    | East   | B  |     319 |     233 |
    | 2020-01-05 | 2020-01-11 | Italy    | East   | D  |     261 |     187 |


    Parameters
    ----------
    start: start year

    end: end year

    freq: frequency (default 'D' days)

    seed: random seed, default 42


    Returns
    -------
    A pandas Dataframe
    '''

    if seed is not None:
        np.random.seed(seed)

    sample_period = pd.date_range(start=str(start), end=str(end), freq=freq)
    len_period = len(sample_period)
    dates = pd.Series(sample_period, name='dates')

    order_year_offset = np.random.randint(1, 10)
    order_dates = (pd.date_range(start=str(start), end=str(end), freq=freq)
                     .shift(freq=freq, periods=np.random.randint(1, 30)))

    values1 = pd.Series(np.random.randint(low=10, high=400, size=len_period), name='values_1')
    values2 = pd.Series(np.random.randint(low=10, high=400, size=len_period), name='values_2')

    id_list = ['A', 'B', 'C', 'D', 'E']
    ids = pd.Series(np.random.choice(id_list, size=len_period), name='ids')

    region_list = ['East', 'West', 'North', 'South']
    regions = pd.Series(np.random.choice(region_list, size=len_period), name='regions')

    country_list = ['Germany', 'Italy', 'France', 'Spain', 'Sweden', 'Portugal', 'Norway', 'Switzerland']
    countries = pd.Series(np.random.choice(country_list, size=len_period), name='countries')

    data_dictionary = {
        'dates': dates,
        'order_dates': order_dates,
        'countries': countries,
        'regions': regions,
        'ids': ids,
        'values_1': values1,
        'values_2': values2
    }

    df = pd.DataFrame(data_dictionary)

    return df


# get_sample_sales {{{1
def get_sample_sales(number_of_rows: int = 200,
                     year: int = 2021,
                     seed: int = 42) -> pd.DataFrame:
    ''' TEST - Generate sales product data

    location, product, month, target_sales,
    target_profit, actual_sales, actual profit.

    Example
    -------
    get_sample_sales().head()

    | location| product   | month     |target_sales |target_profit |actual_sales |actual_profit |
    |:--------|:----------|:----------|------------:|-------------:|------------:|-------------:|
    | London  | Beachwear | 2021-01-01|       31749 |      1904.94 |     29209.1 |      1752.54 |
    | London  | Beachwear | 2021-01-01|       37833 |      6053.28 |     34049.7 |      5447.95 |
    | London  | Jeans     | 2021-01-01|       29485 |      4127.9  |     31549   |      4416.85 |
    | London  | Jeans     | 2021-01-01|       37524 |      3752.4  |     40901.2 |      4090.12 |
    | London  | Sportswear| 2021-01-01|       27216 |      4354.56 |     29121.1 |      4659.38 |


    Parameters
    ----------
    number_of_rows - int - number of records/rows required.

    year - int - sales year to generate data for.

    seed: int - random seed, default 42


    Returns
    -------
    Pandas Dataframe

    '''
    if seed is not None:
        np.random.seed(seed)

    locations = ['London', 'Paris', 'Milan']
    products = ['Tops & Blouses', 'Jeans', 'Footwear',
                'Beachwear', 'Sportswear']

    data = {
        'location': np.random.choice(locations, size=number_of_rows),
        'product': np.random.choice(products, size=number_of_rows),
        'month': np.random.choice(range(1, 13), size=number_of_rows),
        'target_sales': np.random.randint(14000, 40000, size=number_of_rows),
        'target%_profit': np.random.randint(10, size=number_of_rows) * .02
    }

    df = pd.DataFrame(data)

    df['month'] = df['month'].apply(
        lambda x: pd.Period(f'{year}-{str(x).zfill(2)}'))

    df['target_profit'] = df['target_sales'] * df['target%_profit']

    actual_sales_lambda = lambda x: x + (x * np.random.choice(range(-10, 10)) / 100)
    df['actual_sales'] = df['target_sales'].apply(actual_sales_lambda)

    df['actual_profit'] = (df['actual_sales'] * df['target%_profit']).round(2)

    df.drop(columns=['target%_profit'], inplace=True)
    df.month = pd.PeriodIndex(df.month).to_timestamp()

    df = df.sort_values(['month', 'location', 'product'])

    return df


# get_sample_matrix {{{1
def get_sample_matrix(size: Tuple = (5, 5),
                      loc: int = 10,
                      scale: int = 10,
                      lowercase_cols: bool = True,
                      round: int = 3,
                      seed:int = 42) -> pd.DataFrame:
    ''' Generate sample data for given size (tuple)

    Draw random samples from a normal (Gaussian) distribution.
    (Uses np.random.normal)

    Example
    -------
    df = sample_data()

    |    |        a |        b |         c |        d |         e |
    |---:|---------:|---------:|----------:|---------:|----------:|
    |  0 |  28.2635 |  58.5958 | -37.7419  |  43.1493 |  30.6003  |
    |  1 |  68.6657 | 133.374  |  -4.08528 |  30.1512 |  74.854   |
    |  2 |  20.979  | 206.842  |  25.84    |  54.9084 | -40.8005  |
    |  3 |  80.6453 |  98.5271 |  19.2099  |  93.3201 |  -2.92593 |
    |  4 | -31.4728 |  50.5013 | -46.5339  | -27.3093 |  80.5112  |


    Parameters
    ----------
    size: tuple - row and column size required.

    loc : float or array_like of floats
    Mean ("centre") of the distribution.

    scale : float or array_like of floats
    Standard deviation (spread or "width") of the distribution.
    Must be non-negative.

    lowercase_cols: boolean - alphabetical column names are lowercase
    by default. False - uppercase column names


    Returns
    -------
    pd.DataFrame
    '''
    if seed:
        np.random.seed(seed)

    rows, cols = size

    data = np.random.normal(loc=loc, scale=scale, size=(rows, cols))
    cols = [xl_col_to_name(x - 1) for x in range(1, cols + 1)]

    if lowercase_cols:
        cols = list(map(str.lower, cols))

    df = pd.DataFrame(data, columns=cols).round(round)

    return df

# generate_periods {{{1
def generate_periods(year: int = 2021,
                     month_range: Tuple = (1, 12),
                     delta_range: Tuple = (1, 10),
                     rows: int = 20,
                     seed: int = 42) -> pd.DataFrame:
    ''' Generate random effective and expired period pairs

    Usage example
    -------------
    head(generate_periods(year=2022, rows=1))

    effective	expired
    2022-07-20	2022-07-28


    Parameters
    ----------
    year: int - desired year

    month_range: tuple - range of months required for effective date range.count

    delta_range: tuple - range of delta periods to add to effective

    rows: int - number of rows or records needed

    seed: int - default 42 - for testing, provide consistent outcome by
                setting the random seed value

    Returns
    -------
    A pandas dataframe containing generated effective and expired values.
   '''
    def create_date(period):
        '''
        '''

        period_str = period.strftime('%Y-%m')
        day_str = str(np.random.randint(1, period.daysinmonth))

        return f'{period_str}-{day_str.zfill(2)}'

    if seed:
        np.random.seed(seed)

    low, high = month_range
    months = np.random.randint(low, high, size=rows)
    periods = [pd.Period(f'{year}-{str(month).zfill(2)}') for month in months]
    periods = pd.Series(periods, name='Periods')

    effective = pd.Series([pd.to_datetime(create_date(period)) for period in periods], name='effective')

    low, high = delta_range
    deltas = pd.Series(pd.to_timedelta(np.random.randint(low, high, rows), 'd'), name='duration')

    expired = effective + deltas
    expired.name = 'expired'

    df = pd.concat([effective, expired], axis=1)

    return df
