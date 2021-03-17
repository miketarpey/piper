import pandas as pd
import numpy as np
import json
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


# sample_data {{{1
def sample_data(start: int = 2020,
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


# sample_sales {{{1
def sample_sales(number_of_rows: int = 200,
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


# sample_matrix {{{1
def sample_matrix(size: Tuple = (5, 5),
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
    ''' Generate random effective and expired period pair values

    Usage example
    -------------
    head(generate_periods(year=2022, rows=5))

    | effective   | expired    |
    |:------------|:-----------|
    | 2022-07-07  | 2022-07-15 |
    | 2022-04-26  | 2022-05-01 |
    | 2022-11-19  | 2022-11-23 |
    | 2022-08-23  | 2022-08-31 |

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
    A pandas dataframe containing generated effective and expired pair values.
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


# make_null_dates(): {{{1
def make_null_dates(df: pd.DataFrame,
                    cols: Union[str, List[str]] = ['effective', 'expired'],
                    null_values_percent: float = .2,
                    seed: int = 42) -> pd.DataFrame:
    ''' Generate 'random' null, pd.NaT values

    Parameters
    ----------
    df: pandas DataFrame

    cols: column(s) within dataframe to generate random null values for

    null_value_percent: % number of rows to generate null values


    Returns
    -------
    A pandas DataFrame
    '''
    if seed:
        np.random.seed(seed)

    rows = df.shape[0]
    percent_to_make_null = int(rows * null_values_percent)

    for col in cols:

        for row in np.random.randint(1, rows, percent_to_make_null):
            df.loc[row, col] = pd.NaT

    return df


# bad_quality_orders(): {{{1
def bad_quality_orders():
    '''
    Generate a dataset with a number of issues to be 'cleaned'.
    - column names too long
    - invalid numeric data
    - invalid/uneven character data
    - invalid date data

    Returns
    -------
    pandas DataFrame
    '''

    data = ''' {"Gropuing cde_": {"0": "A100", "1": "A101", "2": "A101",
                                  "3": "A102", "4": "A103", "5": "A103",
                                  "6": "A103", "7": "A104", "8": "A105",
                                  "9": "A105", "10": "A102", "11": "A103"},
            "Order_NBR": {"0": 23899001, "1": 23899002, "2": 23899003,
                          "3": 23899004, "4": 23899005, "5": 23899006,
                          "6": 23899007, "7": 23899008, "8": 23899009,
                          "9": 23899010, "10": 23899011, "11": 23899012},
            "This column name is too long":
            {"0": "First                        row", "1": "SECOnd   Row", "2":
             "Thrid        Row", "3": "fOuRth        ROW", "4": "fIFTH rOw",
             "5": "sIxTH        rOw", "6": "SeVENtH roW", "7": "EIghTh RoW",
             "8": "NINTH      row", "9": "TEnTh           rOW",
             "10": "fOuRth      Row", "11": "fIFTH      rOw"},
            "Second column ": {"0": "Scally, Aidan", "1": "McAllister, Eoin",
                               "2": "Tarpey, Mike", "3": "Denton,        Alan",
                               "4": "Dallis,                  Theo",
                               "5": "HUNT, DerEK",
                               "6": "Goddard,     TONY",
                               "7": "Whitaker,    Matthew",
                               "8": "Seiffert, CARsteN",
                               "9": "Freer,        Craig",
                               "10": "DENTON,   Alan",
                               "11": "Dallis,           THEO"},
            "Quantity": {"0": 14, "1": 103, "2": 1, "3": 13, "4": 19,
                         "5": "5---32,14", "6": 178.3035, "7": "4-2-4,00",
                         "8": "2-4,00", "9": "18,00.22", "10": 13, "11": 19},
            "Price": { "0": "          1,23", "1": "      4,32",
                       "2": "3  4,32", "3": 49, "4": "45.7,98",
                       "5": "634,23", "6": 27.04502, "7": "56,3.00",
                       "8": "9,.8.00", "9": 563, "10": 49, "11": "45.7,98"},
            "Effective": {"0": "21.10.2015", "1": "21.10.2016",
                          "2": "21.10.2017", "3": "21.10.2018",
                          "4": "21.10.2019", "5": "21\\/04\\/2019",
                          "6": "21.10.2019", "7": "21.10.2019",
                          "8": "21-5-2018", "9": "21.10.2019",
                          "10": "21.10.2018", "11": "2019.10.21"},
            "Expired": {"0": "31.12.2019", "1": "31.12.2020",
                        "2": "31.12.2021", "3": "31.12.2022",
                        "4": "31.12.2023", "5": "05.07.2020",
                        "6": "31\\/12\\/2025", "7": "31\\/12\\/2025",
                        "8": "2025.12.10", "9": "31\\/12\\/2025",
                        "10": "2022.12.31", "11": "31.12.2023"},
            "TranSACTion DATE": {"0": "20.08.2018", "1": "20.08.2017",
                                 "2": "20.08.2020", "3": "20.08.2021",
                                 "4": "20.08.2022", "5": "20.08.2015",
                                 "6": "20.08.2024", "7": "20.08.2025",
                                 "8": "20.08.2026", "9": "20.08.2027",
                                 "10": "20.08.2021", "11": "20.08.2022"}
    }'''

    df = pd.DataFrame(json.loads(data))

    return df


# two_columns_five_rows {{{1
def two_columns_five_rows():

    id_list = ['A', 'B', 'C', 'D', 'E']
    s1 = pd.Series(np.random.choice(id_list, size=5), name='ids')

    region_list = ['East', 'West', 'North', 'South']
    s2 = pd.Series(np.random.choice(region_list, size=5), name='regions')

    df = pd.concat([s1, s2], axis=1)

    return df


# xl_test_data() {{{1
def xl_test_data():
    '''
    Generate a dataset with valid data for testing

    Returns
    -------
    pandas DataFrame
    '''

    data = ''' {"Customer": {"0": "1042083", "1": "1042084", "2": "1042085",
                             "3": "1042086", "4": "1042087", "5": "1042088",
                             "6": "1042089", "7": "1042090", "8": "1042091",
                             "9": "A1042092", "10": "1042093",
                             "11": "1042094"},
                "Target_name": {"0": "Customer1", "1": "Customer2",
                                "2": "Customer   3", "3": "Customer4",
                                "4": "Customer5", "5": "Customer6",
                                "6": "Customer7", "7": "Customer8",
                                "8": "Customer9", "9": "Customer10",
                                "10": "Customer11", "11": "Customer12"},
                "Public": {"0": "Y", "1": "Y", "2": "Y", "3": "Y", "4": "Y",
                           "5": "Y", "6": "Y", "7": "A", "8": "Y", "9": "Y",
                           "10": "Y", "11": "Y"},
                "Material": {"0": "GLB0001", "1": "GLB0002", "2": "GLB0003",
                             "3": "GLB0004", "4": "GLB0005", "5": "GLB  0006",
                             "6": "GLB0007", "7": "GLB0008", "8": "GLB0009",
                             "9": "GLB0010", "10": "GLB0011", "11": "GLB0012"},
                "Prod_type": {"0": "SP", "1": "FG", "2": "FG", "3": "FG",
                              "4": "FG", "5": "ZY", "6": "FG", "7": "FG",
                              "8": "FG", "9": "FG", "10": "DX", "11": "FG"},
               "Effective": {"0": "2011-03-01", "1": "2011-03-01",
                             "2": "2010-10-07", "3": "2010-10-07",
                             "4": "2011-03-01", "5": "2011-03-01",
                             "6": "2016-01-01", "7": "2016-01-01",
                             "8": "2016-01-01", "9": "2016-01-01",
                             "10": "2015-01-01", "11": "2013-07-16"},
               "Expired": {"0": "2018-03-31", "1": "2018-03-31",
                           "2": "2017-12-31", "3": "2017-12-31",
                           "4": "2018-03-31", "5": "2018-03-31",
                           "6": "2016-12-31", "7": "2016-12-31",
                           "8": "2016-12-31", "9": "2016-12-31",
                           "10": "2016-12-31", "11": "2016-10-31"},
               "Price": {"0": 408, "1": 408, "2": 486.48, "3": 489.48,
                         "4": 528, "5": 528, "6": 558.72, "7": 558.72,
                         "8": 558.72, "9": 558.72, "10": 2265.6, "11": 8.57},
               "Currency": {"0": "EUR", "1": "EUR", "2": "EUR", "3": "EUR",
                            "4": "AUD", "5": "EUR", "6": "EUR", "7": "EUR",
                            "8": "EUR", "9": "EUR", "10": "EUR", "11": "EUR"}
                            }'''

    df = pd.DataFrame(json.loads(data))

    return df


# simple_series_01 {{{1
def simple_series_01():
    ''' Simple series, one column of alphabetical values '''
    np.random.seed(42)

    id_list = ['A', 'B', 'C', 'D', 'E']
    s1 = pd.Series(np.random.choice(id_list, size=5), name='ids')

    return s1


# get_sample_df3 {{{1
def get_sample_df3():

    id_list = ['A', 'B', 'C', 'D', 'E']
    s1 = pd.Series(np.random.choice(id_list, size=5), name='ids')

    region_list = ['East', 'West', 'North', 'South']
    s2 = pd.Series(np.random.choice(region_list, size=5), name='regions')

    region_list = ['UK', 'France', 'Germany', 'Italy']
    s3 = pd.Series(np.random.choice(region_list, size=5), name='countries')

    df = pd.concat([s1, s2, s3], axis=1)

    return df


# get_sample_df4 {{{1
def get_sample_df4():

    dict_a = {'column_A': {'0': 'A100',  '1': 'A101',  '2': 'A101',
                           '3': 'A102',  '4': 'A103',  '5': 'A103',
                           '6': 'A103',  '7': 'A104',  '8': 'A105',
                           '9': 'A105', '10': 'A102', '11': 'A103'}}

    df = pd.DataFrame(dict_a)

    dict_b = {'column_B': {'0': 'First Row',  '1': 'Second Row',
                           '2': 'Fourth Row', '3': 'Fifth Row',
                           '4': 'Third Row',  '5': 'Fourth Row',
                           '6': 'Fifth Row',   '7': 'Sixth Row',
                           '8': 'Seventh Row', '9': 'Eighth Row',
                           '10': 'Ninth Row', ' 11': 'Tenth Row'}}
    df2 = pd.DataFrame(dict_b)

    return df, df2


# get_sample_df5 {{{1
def get_sample_df5():

    test = ''' {"bgy56icnt":{"0":"BE","1":"BE","2":"BE","3":"BE","4":"BE"},
                "bgz56ccode":{"0":46065502,"1":46065502,"2":46065502,
                "3":46065798,"4":46066013}} '''

    df = pd.DataFrame(json.loads(test))

    return df


# get_sample_df6 {{{1
def get_sample_df6():

    test = ''' {"column_A":{"0":"AA","1":"BB","2":"CC","3":"DD","4":"EE"},
                "column_B":{"0":100,"1":200,"2":300,"3":400,"4":500}} '''
    df = pd.DataFrame(json.loads(test))

    test = ''' {"column_A":{"0":"A_","1":"AA","2":"AA","3":"BB","4":"CC"},
                "column_B":{"0":100,"1":200,"2":300,"3":400,"4":500}} '''
    df2 = pd.DataFrame(json.loads(test))

    return df, df2


# single_column_dataframe_messy_text {{{1
def single_column_dataframe_messy_text():

    test = ''' {"test_col": {
            "0": "First                        row",
            "1": "SECOnd   Row ",
            "2": "Thrid        Row          ",
            "3": "fOuRth        ROW  ",
            "4": "fIFTH rOw",
            "5": "sIxTH        rOw      ",
            "6": "SeVENtH roW     ",
            "7": "EIghTh RoW     ",
            "8": "NINTH      row",
            "9": "TEnTh           rOW         ",
            "10": "fOuRth      Row          ",
            "11": "fIFTH      rOw         "
        } }'''
    df = pd.DataFrame(json.loads(test))

    return df
