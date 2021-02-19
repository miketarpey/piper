import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_sample_data(start='2020', end='2021', freq='D', seed=30):
    ''' NOTE:: This function for use when testing in notebooks
                - nothing to do with PYTESTING

    Example
    -------
    df = get_sample_data()


    Parameters
    ----------
    start: int - start year

    end: int - end year

    freq: str - frequency (default 'D' days)

    seed: int - random seed, default 42


    Returns
    -------
    Pandas Dataframe
    '''

    if seed is not None:
        np.random.seed(seed)

    sample_period = pd.date_range(start=start, end=end, freq=freq)
    len_period = len(sample_period)
    dates = pd.Series(sample_period, name='dates')

    order_year_offset = np.random.randint(1, 10)
    order_dates = (pd.date_range(start=start, end=end, freq=freq)
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


def get_sample_sales(number_of_rows = 200, year=2021):
    '''
    Generate sales product test data for given year:

    location, product, month, target_sales,
    target_profit, actual_sales, actual profit.

    Example
    -------
    df = get_sample_sales()


    Parameters
    ----------
    number_of_rows - int - number of records/rows required.

    year - int - sales year to generate data for.

    seed: int - random seed, default 42


    Returns
    -------
    Pandas Dataframe

    '''

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
