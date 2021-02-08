import pytest
import json
import pandas as pd
import numpy as np


@pytest.fixture
def get_sample_orders_01():
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


@pytest.fixture
def get_sample_orders_02():
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


@pytest.fixture
def get_sample_df1(start='2020', end='2021', freq='D', seed=30):
    ''' Generate and return sample time series data
    '''

    if seed is not None:
        np.random.seed(seed)

    sample_period = pd.date_range(start=start, end=end, freq=freq)
    len_period = len(sample_period)
    dates = pd.Series(sample_period, name='dates')

    order_year_offset = np.random.randint(1, 10)
    order_dates = (pd.date_range(start=start, end=end, freq=freq)
                     .shift(freq=freq, periods=np.random.randint(1, 30)))

    values_1 = pd.Series(np.random.randint(low=10, high=400, size=len_period), name='values_1')
    values_2 = pd.Series(np.random.randint(low=10, high=400, size=len_period), name='values_2')

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
        'values_1': values_1,
        'values_2': values_2
    }

    df = pd.DataFrame(data_dictionary)

    return df


@pytest.fixture
def get_sample_df2():

    id_list = ['A', 'B', 'C', 'D', 'E']
    s1 = pd.Series(np.random.choice(id_list, size=5), name='ids')

    region_list = ['East', 'West', 'North', 'South']
    s2 = pd.Series(np.random.choice(region_list, size=5),
                   name='regions')

    df = pd.concat([s1, s2], axis=1)

    return df


@pytest.fixture
def get_sample_df3():

    id_list = ['A', 'B', 'C', 'D', 'E']
    s1 = pd.Series(np.random.choice(id_list, size=5), name='ids')

    region_list = ['East', 'West', 'North', 'South']
    s2 = pd.Series(np.random.choice(region_list, size=5), name='regions')

    region_list = ['UK', 'France', 'Germany', 'Italy']
    s3 = pd.Series(np.random.choice(region_list, size=5), name='countries')

    df = pd.concat([s1, s2, s3], axis=1)

    return df


@pytest.fixture
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


@pytest.fixture
def get_sample_df5():

    test = ''' {"bgy56icnt":{"0":"BE","1":"BE","2":"BE","3":"BE","4":"BE"},
                "bgz56ccode":{"0":46065502,"1":46065502,"2":46065502,
                "3":46065798,"4":46066013}} '''

    df = pd.DataFrame(json.loads(test))

    return df


@pytest.fixture
def get_sample_df6():

    test = ''' {"column_A":{"0":"AA","1":"BB","2":"CC","3":"DD","4":"EE"},
                "column_B":{"0":100,"1":200,"2":300,"3":400,"4":500}} '''
    df = pd.DataFrame(json.loads(test))

    test = ''' {"column_A":{"0":"A_","1":"AA","2":"AA","3":"BB","4":"CC"},
                "column_B":{"0":100,"1":200,"2":300,"3":400,"4":500}} '''
    df2 = pd.DataFrame(json.loads(test))

    return df, df2


@pytest.fixture
def get_sample_df7():

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
