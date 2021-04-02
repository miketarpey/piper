from piper.custom import to_julian
from piper.factory import dummy_dataframe
from piper.factory import sample_column_clean_text
from piper.factory import sample_data
from piper.factory import sample_sales
from piper.factory import simple_series
from piper.verbs import across
from piper.verbs import adorn
from piper.verbs import assign
from piper.verbs import clean_columns
from piper.verbs import columns
from piper.verbs import combine_header_rows
from piper.verbs import count
from piper.verbs import distinct
from piper.verbs import drop
from piper.verbs import drop_columns
from piper.verbs import duplicated
from piper.verbs import explode
from piper.verbs import flatten_cols
from piper.verbs import fmt_dateidx
from piper.verbs import group_by
from piper.verbs import head
from piper.verbs import info
from piper.verbs import inner_join
from piper.verbs import left_join
from piper.verbs import non_alpha
from piper.verbs import order_by
from piper.verbs import outer_join
from piper.verbs import overlaps
from piper.verbs import pivot_table
from piper.verbs import relocate
from piper.verbs import rename
from piper.verbs import rename_axis
from piper.verbs import right_join
from piper.verbs import sample
from piper.verbs import select
from piper.verbs import set_columns
from piper.verbs import str_trim
from piper.verbs import summarise
from piper.verbs import summary_df
from piper.verbs import split_dataframe
from piper.verbs import str_split
from piper.verbs import str_join
from piper.verbs import tail
from piper.verbs import transform
from piper.verbs import where
from pandas.api.types import is_float_dtype
from pandas._testing import assert_frame_equal
from pandas._testing import assert_series_equal
import numpy as np
import pandas as pd
import pytest
import random


# t_sample_sales {{{1
@pytest.fixture
def t_sample_sales():
    return sample_sales()


# t_sample_data {{{1
@pytest.fixture
def t_sample_data():
    return sample_data()


# t_sample_column_clean_text {{{1
@pytest.fixture
def t_sample_column_clean_text():
    return sample_column_clean_text()


# t_dummy_dataframe {{{1
@pytest.fixture
def t_dummy_dataframe():
    return dummy_dataframe()


# t_simple_series_01 {{{1
@pytest.fixture
def t_simple_series_01():
    return simple_series()


# get_column_list {{{1
@pytest.fixture
def get_column_list():
    """ Column list for dataframe
    """
    column_list = ['dupe**', 'Customer   ', 'mdm no. to use',
                   'Target-name   ', '     Public', '_  Material',
                   'Prod type', '#Effective     ', 'Expired',
                   'Price%  ', 'Currency$']

    return column_list


# test_across_single_column_series_object_function {{{1
def test_across_single_column_series_object_function(t_sample_data):

    df = t_sample_data

    df = across(df, columns='values_1',
                function= lambda x: x.astype(float),
                series_obj=True)

    assert pd.api.types.is_float_dtype(df.values_1)


# test_across_list_column_series_object_function {{{1
def test_across_list_column_series_object_function(t_sample_data):

    df = t_sample_data

    df = across(df, columns=['values_1'],
                function= lambda x: x.astype(float),
                series_obj=True)

    assert pd.api.types.is_float_dtype(df.values_1)


# test_across_list_column_not_series_object {{{1
def test_across_list_column_not_series_object(t_sample_data):

    df = t_sample_data

    df = across(df, columns=['order_dates', 'dates'],
                function=lambda x: to_julian(x), series_obj=False)

    assert df.loc[360, 'dates'] == 120361


# test_across_list_column_not_series_raise_error {{{1
def test_across_list_column_not_series_raise_error(t_sample_data):

    df = t_sample_data

    with pytest.raises(ValueError):
        df = across(df, columns=['order_dates', 'dates'],
                    function=lambda x: to_julian(x), series_obj=True)


# test_across_list_column_series_values_raise_attr_error {{{1
def test_across_list_column_series_values_raise_attr_error(t_sample_data):

    df = t_sample_data

    with pytest.raises(AttributeError):
        across(df, columns=['values_1', 'values_2'],
                function=lambda x: x.astype(int), series_obj=False)


# test_adorn_row_total {{{1
def test_adorn_row_total(t_sample_data):

    df = t_sample_data
    df = group_by(df, ['countries'])
    df = summarise(df, total=('values_1', 'sum'))
    df = adorn(df)

    expected = 73604
    actual = df.loc['All'].values[0]

    assert expected == actual


# test_adorn_column_total {{{1
def test_adorn_column_total(t_sample_data):

    df = t_sample_data
    df = group_by(df, ['countries'])
    df = summarise(df, total=('values_1', 'sum'))
    df = adorn(df, axis = 'column')

    expected = 8432
    actual = df.loc['Sweden', 'All']

    assert expected == actual


# test_adorn_with_ignore_row_index {{{1
def test_adorn_row_with_ignore_row_index(t_sample_data):

    df = t_sample_data
    df = group_by(df, ['countries'])
    df = summarise(df, total=('values_1', 'sum')).reset_index()
    df = adorn(df, axis = 'row', ignore_index=True)

    expected = 'All'
    actual = df.iloc[df.shape[0]-1, 0]

    assert expected == actual


# test_adorn_column_with_column_specified {{{1
def test_adorn_column_with_column_specified(t_sample_data):

    df = t_sample_data
    df = group_by(df, ['countries'])
    df = summarise(df, total=('values_1', 'sum'))
    df = adorn(df, columns='total', axis = 'column')

    expected = 8432
    actual = df.loc['Sweden', 'All']

    assert expected == actual

# test_adorn_column_with_column_list_specified {{{1
def test_adorn_column_with_column_list_specified(t_sample_data):

    df = t_sample_data
    df = group_by(df, ['countries', 'regions'])
    df = summarise(df, total=('values_1', 'sum'))
    df = assign(df, total2=lambda x: x.total * 10)
    df = adorn(df, columns=['total', 'total2'], axis = 'both')

    expected = ['total', 'total2', 'All']
    actual = df.columns.tolist()

    assert expected == actual

# test_adorn_column_with_column_str_specified {{{1
def test_adorn_column_with_column_str_specified(t_sample_data):

    df = t_sample_data
    df = group_by(df, ['countries', 'regions'])
    df = summarise(df, total=('values_1', 'sum'))
    df = assign(df, total2=lambda x: x.total * 10)
    df = adorn(df, columns='total', axis = 'both')

    expected = ['total', 'total2', 'All']
    actual = df.columns.tolist()

    assert expected == actual


# test_assign {{{1
def test_assign(t_sample_data):
    """
    """
    df = t_sample_data
    df = where(df, "ids == 'A'")
    df = where(df, "values_1 > 300 & countries.isin(['Italy', 'Spain'])")
    df = assign(df, new_field=lambda x: x.countries.str[:3]+x.regions,
                       another=lambda x:3*x.values_1)

    expected = ['dates', 'order_dates', 'countries',
                'regions', 'ids', 'values_1',
                'values_2', 'new_field', 'another']
    actual = df.columns.tolist()

    assert expected == actual


# test_assign_with_dataframe_object_formulas {{{1
def test_assign_with_dataframe_object_formulas(t_sample_data):
    """
    """
    df = t_sample_data
    df = where(df, "ids == 'A'")
    df = where(df, "values_1 > 300 & countries.isin(['Italy', 'Spain'])")
    df = assign(df, new_field=lambda x: x.countries.str[:3] + x.regions,
                    another=lambda x: 3*x.values_1)

    expected = ['dates', 'order_dates', 'countries', 'regions', 'ids', 'values_1',
                'values_2', 'new_field', 'another']
    actual = df.columns.tolist()

    assert expected == actual

# test_assign_with_tuple_function {{{1
def test_assign_with_tuple_function(t_sample_data):

    df = t_sample_data
    df = assign(df, reversed=('regions', lambda x: x[::-1]))
    df = select(df, ['-dates', '-order_dates'])

    expected = ['countries', 'regions', 'ids', 'values_1', 'values_2', 'reversed']
    actual = df.columns.tolist()

    assert expected == actual


# test_assign_with_value_error {{{1
def test_assign_with_value_error(t_sample_data):

    df = t_sample_data

    with pytest.raises(ValueError):
        actual = assign(df, reversed=lambda x: x[::-1])


# test_clean_column_list_using_list_and_title {{{1
def test_clean_column_list_using_list_and_title(get_column_list):
    """
    """
    expected = ['Dupe', 'Customer', 'Mdm_No_To_Use', 'Target_Name', 'Public',
                'Material', 'Prod_Type', 'Effective', 'Expired',
                'Price', 'Currency']

    dx = pd.DataFrame(None, columns=get_column_list)

    actual = clean_columns(dx, title=True).columns.tolist()
    assert expected == actual

# test_clean_column_list_using_df_columns {{{1
def test_clean_column_list_using_df_columns(get_column_list):
    """
    """
    expected = ['dupe', 'customer', 'mdm_no_to_use', 'target_name', 'public',
                'material', 'prod_type', 'effective', 'expired',
                'price', 'currency']

    dx = pd.DataFrame(None, columns=get_column_list)

    actual = clean_columns(dx).columns.tolist()
    assert expected == actual


# test_columns_as_list {{{1
def test_columns_dataframe(t_sample_data):

    df = t_sample_data

    expected = ['dates', 'order_dates', 'countries', 'regions', 'ids', 'values_1', 'values_2']

    actual = columns(df, astype='list')
    assert expected == actual

    expected = {'ids': 'ids', 'regions': 'regions'}
    actual = columns(df[['ids', 'regions']], astype='dict')
    assert expected == actual

    expected = pd.DataFrame(['ids', 'regions'], columns=['column_names'])
    actual = columns(df[['ids', 'regions']], astype='dataframe')
    assert_frame_equal(expected, actual)

    expected = "['ids', 'regions']"
    actual = columns(df[['ids', 'regions']], astype='text')
    assert expected == actual


# test_columns_as_series {{{1
def test_columns_as_series(t_sample_data):

    df = t_sample_data

    cols = ['dates', 'order_dates', 'countries', 'regions', 'ids', 'values_1', 'values_2']
    expected = pd.Series(cols, index=range(len(cols)), name='column_names')
    actual = columns(df, astype='series')

    assert_series_equal(expected, actual)


# test_columns_as_regex {{{1
def test_columns_as_regex(t_sample_data):

    prices = {
        'prices': [100, 200, 300],
        'contract': ['A', 'B', 'A'],
        'effective': ['2020-01-01', '2020-03-03', '2020-05-30'],
        'expired': ['2020-12-31', '2021-04-30', '2022-04-01']
    }

    df = pd.DataFrame(prices)

    expected = ['effective', 'expired']
    actual = columns(df, regex='e', astype='list')

    assert expected == actual


# test_combine_header_rows {{{1
def test_combine_header_rows_with_title():

    data = {'A': ['Order', 'Qty', 10, 40],
            'B': ['Order', 'Number', 12345, 12346]}

    df = pd.DataFrame(data)
    df = combine_header_rows(df, title=True)

    expected = ['Order Qty', 'Order Number']
    actual = df.columns.to_list()

    assert expected == actual


# test_combine_header_rows_lower {{{1
def test_combine_header_rows_lower():

    data = {'A': ['Order', 'Qty', 10, 40],
            'B': ['Order', 'Number', 12345, 12346]}

    df = pd.DataFrame(data)
    df = combine_header_rows(df, title=False)

    expected = ['order qty', 'order number']
    actual = df.columns.to_list()

    assert expected == actual


# test_combine_header_rows_title {{{1
def test_combine_header_rows_title_bad_data():

    data = {'A': ['Order   ', 'Qty   ', 10, 40],
            'B': [' Order', '    Number%', 12345, 12346]}

    df = pd.DataFrame(data)
    df = combine_header_rows(df, title=True)

    expected = ['Order Qty', 'Order Number']
    actual = df.columns.to_list()

    assert expected == actual


# test_count_series{ {{1
def test_count_series(t_simple_series_01):

    s1 = t_simple_series_01

    expected = (3, 3)
    actual = count(s1).shape

    assert expected == actual


# test_count_sort {{{1
def test_count_sort(t_simple_series_01):

    s1 = t_simple_series_01

    expected = 3

    count_ = count(s1, sort_values=False)
    actual = count_.loc['E', 'n']

    assert expected == actual

# test_count_with_total {{{1
def test_count_with_total(t_simple_series_01):

    s1 = t_simple_series_01

    expected = 100.0
    count_ = count(s1, totals=True)
    actual = count_.loc['Total', '%']

    assert expected == actual


# test_count_with_total_percent_cum_percent {{{1
def test_count_with_total_percent_cum_percent(t_simple_series_01):

    s1 = t_simple_series_01

    expected = (4, 3)
    actual = count(s1, totals=True, sort_values=True,
                   percent=True, cum_percent=True).shape

    assert expected == actual


# test_count_with_cum_percent_with_threshold {{{1
def test_count_with_cum_percent_with_threshold(t_simple_series_01):

    s1 = t_simple_series_01

    expected = (2, 3)

    count_ = count(s1, threshold=81, cum_percent=True)
    actual = count_.shape

    assert expected == actual


# test_count_not_found_column {{{1
def test_count_not_found_column(t_sample_data):

    df = t_sample_data

    expected = None
    actual = count(df, 'invalid_column')

    assert expected == actual


# test_count_no_column {{{1
def test_count_no_column(t_sample_data):

    df = t_sample_data

    expected = (7, 3)
    actual = count(df).shape

    assert expected == actual


# test_count_column_reset_index_true {{{1
def test_count_column_reset_index_true(t_sample_data):

    df = t_sample_data

    expected = (8, 4)
    actual = count(df, 'countries', reset_index=True).shape

    assert expected == actual


# test_count_single_column {{{1
def test_count_single_column(t_sample_data):

    df = t_sample_data

    expected = (4, 3)
    actual = count(df, 'regions').shape

    assert expected == actual


# test_count_multi_column {{{1
def test_count_multi_column(t_sample_data):

    df = t_sample_data

    query = "regions == 'East' and countries.isin(['Italy'])"
    df = df.query(query)

    expected = (1, 5)
    actual = count(df, ['regions', 'countries']).reset_index().shape

    assert expected == actual


# test_count_multi_column_with_percent {{{1
def test_count_multi_column_with_percent(t_sample_data):

    df = t_sample_data
    df = df.query("regions == 'East' and countries.isin(['Italy'])")
    df = count(df, ['regions', 'countries'], percent=True, cum_percent=True)

    expected = 100.0
    actual = df.loc[('East', 'Italy'), 'cum %']

    assert expected == actual


# test_count_multi_column_with_cum_percent_threshold {{{1
def test_count_multi_column_with_cum_percent_threshold(t_sample_data):

    df = t_sample_data
    df = df.query("regions == 'East'")
    df = count(df, ['regions', 'countries'],
               percent=True,
               cum_percent=True,
               threshold=81)

    expected = (5, 3)
    actual = df.shape

    assert expected == actual


# test_distinct {{{1
def test_distinct(t_sample_data):

    df = t_sample_data
    df = select(df, ['countries', 'regions', 'ids'])
    df = distinct(df, 'ids', shape=True)

    expected = (5, 3)
    actual = df.shape

    assert expected == actual


# test_drop {{{1
def test_drop(t_sample_data):
    """
    """
    df = t_sample_data
    df = drop(df, columns=['countries', 'regions'])

    expected = ['dates', 'order_dates', 'ids', 'values_1', 'values_2']
    actual = df.columns.tolist()

    assert expected == actual


# test_drop_columns {{{1
def test_drop_columns(t_dummy_dataframe):
    """
    """
    df = t_dummy_dataframe

    expected = (5, 5)
    actual = drop_columns(df).shape

    assert expected == actual


# test_duplicated {{{1
def test_duplicated(t_simple_series_01):
    """
    """
    df = t_simple_series_01.to_frame()
    df = duplicated(df, keep=False, sort=True)

    expected = 3  # Duplicate records
    actual = df.duplicate.value_counts()[1]

    assert expected == actual


# test_duplicated_duplicates_only {{{1
def test_duplicated_duplicates_only(t_simple_series_01):
    """
    """
    df = t_simple_series_01.to_frame()

    df = t_simple_series_01.to_frame()
    df = duplicated(df, keep='first', duplicates=True, sort=True)

    expected = (2, 2)
    actual = df.shape

    assert expected == actual


# test_explode {{{1
def test_explode(t_sample_data):
    """
    """
    df = t_sample_data
    df = group_by(df, 'countries')
    df = summarise(df, ids=('ids', set))

    expected = (40, 1)
    actual = explode(df, 'ids').shape

    assert expected == actual


# test_flatten_cols_no_index {{{1
def test_flatten_cols_no_index(t_sample_data):
    """
    """
    df = t_sample_data
    df = group_by(df, ['countries', 'regions'])
    df = summarise(df, total=('values_2', 'sum'))
    df = df.unstack()
    df = df.reset_index()

    # Twice called is deliberate NOT a mistake :)
    # The second call is to make sure function does
    # not crash, just returns passed given column names
    df = flatten_cols(df, remove_prefix='total')
    df = flatten_cols(df, remove_prefix='total')

    expected = ['countries', 'East', 'North', 'South', 'West']
    actual = df.columns.to_list()

    assert expected == actual


# test_flatten_cols_keep_prefix {{{1
def test_flatten_cols_keep_prefix(t_sample_data):
    """
    """
    df = t_sample_data
    df = group_by(df, ['countries', 'regions'])
    df = summarise(df, total=('values_2', 'sum'))
    df = df.unstack()
    df = flatten_cols(df)

    expected = ['total_East', 'total_North', 'total_South', 'total_West']
    actual = df.columns.to_list()

    assert expected == actual


# test_flatten_cols_lose_prefix {{{1
def test_flatten_cols_lose_prefix(t_sample_data):
    """
    """
    df = t_sample_data
    df = group_by(df, ['countries', 'regions'])
    df = summarise(df, total=('values_2', 'sum'))
    df = df.unstack()
    df = flatten_cols(df, remove_prefix='total')

    expected = ['East', 'North', 'South', 'West']
    actual = df.columns.to_list()

    assert expected == actual


# test_flatten_cols_remove_prefix {{{1
def test_flatten_cols_remove_prefix(t_sample_data):
    """
    """
    df = t_sample_data
    df = group_by(df, ['countries', 'regions'])
    df = summarise(df, total=('values_2', 'sum'))
    df = df.unstack()
    df = flatten_cols(df, remove_prefix='total')

    expected = ['East', 'North', 'South', 'West']
    actual = df.columns.to_list()

    assert expected == actual


# test_split_dataframe {{{1
def test_split_dataframe(t_sample_data):

    dataframes = split_dataframe(sample_data(), chunk_size=100)

    expected = 367
    actual = sum([df.shape[0] for df in dataframes])

    assert expected == actual


# test_summary_df {{{1
def test_generate_summary_df():
    """
    """
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

    datasets = [('1st dataset', df), ('2nd dataset', df2)]

    summary = summary_df(datasets, title='Summary',
                            col_total='Total records',
                            add_grand_total=True,
                            grand_total='Grand total')

    expected = 24
    actual = summary.loc['Grand total', 'Total records']

    assert expected == actual

# test_has_special_chars {{{1
def test_non_alpha(t_sample_data):
    """
    """
    np.random.seed(42)

    alphabet="0123456789abcdefghijklmnopqrstuvwxyz!$%()"

    word_list = []
    for _ in range(20):
        chars = np.random.choice(list(alphabet), size=np.random.randint(1, 10))
        word_list.append(''.join(chars))

    df = pd.DataFrame(word_list, columns=['words'])
    df = non_alpha(df, 'words')
    df_count = count(df, 'non_alpha')

    assert df_count.shape == (2, 3)


# test_head_with_series {{{1
def test_head_with_series(t_simple_series_01):
    """
    """
    s1 = t_simple_series_01

    expected = (4,)
    actual = head(s1).shape
    assert expected == actual


# test_head_with_dataframe {{{1
def test_head_with_dataframe(t_sample_data):
    """
    """
    df = t_sample_data

    expected = (4, 7)
    actual = head(df).shape
    assert expected == actual


# test_head_with_columns_function {{{1
def test_head_with_columns_function(t_sample_data):
    """
    """
    df = t_sample_data

    expected = (4, 4)

    actual = head(df[columns(df, regex='order|dates|values',
        astype='list')]).shape

    assert expected == actual

# test_info {{{1
def test_info(t_simple_series_01):

    df = t_simple_series_01.to_frame()

    expected = (1, 7)
    actual = info(df).shape

    assert expected == actual

# test_info_with_dupes {{{1
def test_info_with_dupes(t_simple_series_01):

    df = t_simple_series_01.to_frame()

    expected = (1, 8)
    actual = info(df, n_dupes=True).shape

    assert expected == actual

# test_info_with_na_cols {{{1
def test_info_with_na_cols(t_simple_series_01):

    df = t_simple_series_01.to_frame()

    expected = (0, 7)
    actual = info(df, fillna=True).shape

    assert expected == actual

# test_order_by_single_col_ascending {{{1
def test_order_by_single_col_ascending():

    prices = {'prices': [100, 200, 300],
              'contract': ['A', 'B', 'A'],
              'effective': ['2020-01-01', '2020-03-03', '2020-05-30'],
              'expired': ['2020-12-31', '2021-04-30', '2022-04-01']}

    exp = 200

    got = (pd.DataFrame(prices)
             .pipe(group_by, 'contract')
             .pipe(summarise, prices=('prices', 'sum'))
             .pipe(order_by, by='prices').iloc[0, 0])

    assert exp == got


# test_order_by_single_col_descending {{{1
def test_order_by_single_col_descending():

    prices = {'prices': [100, 200, 300],
              'contract': ['A', 'B', 'A'],
              'effective': ['2020-01-01', '2020-03-03', '2020-05-30'],
              'expired': ['2020-12-31', '2021-04-30', '2022-04-01']}

    exp = 400

    got = (pd.DataFrame(prices)
             .pipe(group_by, 'contract')
             .pipe(summarise, prices=('prices', 'sum'))
             .pipe(order_by, '-prices').iloc[0, 0])

    assert exp == got


# test_order_by_multi_col_descending {{{1
def test_order_by_multi_col_descending(t_sample_sales):

    exp = 404440.24

    df = (t_sample_sales
          .pipe(group_by, ['location', 'product'])
          .pipe(summarise, TotalSales=('actual_sales', 'sum'))
          .pipe(order_by, ['location', '-TotalSales']))

    got = df.loc[('London', slice(None)), 'TotalSales'][0]

    assert exp == got


# test_order_by_multi_col_ascending_by_keyword {{{1
def test_order_by_multi_col_ascending_by_keyword(t_sample_sales):

    exp = 274674.0

    df = (t_sample_sales
          .pipe(group_by, ['location', 'product'])
          .pipe(summarise, TotalSales=('actual_sales', 'sum'))
          .pipe(order_by, by=['location', 'TotalSales']))

    got = df.loc[('London', slice(None)), 'TotalSales'][0]

    assert exp == got


# test_order_by_multi_col_ascending_without_keyword {{{1
def test_order_by_multi_col_ascending_without_keyword(t_sample_sales):

    exp = 274674.0

    df = (t_sample_sales
          .pipe(group_by, ['location', 'product'])
          .pipe(summarise, TotalSales=('actual_sales', 'sum'))
          .pipe(order_by, ['location', 'TotalSales']))

    got = df.loc[('London', slice(None)), 'TotalSales'][0]

    assert exp == got


# test_overlaps {{{1
def test_overlaps():

    prices = {'prices': [100, 200, 300],
              'contract': ['A', 'B', 'A'],
              'effective': ['2020-01-01', '2020-03-03', '2020-05-30'],
              'expired': ['2020-12-31', '2021-04-30', '2022-04-01']}

    df = pd.DataFrame(prices)

    expected = (3, 5)
    actual = overlaps(df, start='effective', end='expired', unique_key='contract')

    assert expected == actual.shape


# test_overlaps_raises_key_error {{{1
def test_overlaps_raises_key_error():

    prices = {'prices': [100, 200, 300],
              'contract': ['A', 'B', 'A'],
              'effective': ['2020-01-01', '2020-03-03', '2020-05-30'],
              'expired': ['2020-12-31', '2021-04-30', '2022-04-01']}

    df = pd.DataFrame(prices)

    with pytest.raises(KeyError):
        actual = overlaps(df, start='false_field', end='expired', unique_key='contract')


# test_overlaps_no_price {{{1
def test_overlaps_unique_key_list():

    prices = {'prices': [100, 200, 300],
              'contract': ['A', 'B', 'A'],
              'effective': ['2020-01-01', '2020-03-03', '2020-05-30'],
              'expired': ['2020-12-31', '2021-04-30', '2022-04-01']}

    df = pd.DataFrame(prices)

    expected = (3, 5)
    actual = overlaps(df, start='effective', end='expired', unique_key=['contract'])

    assert expected == actual.shape


# test_pivot_table {{{1
def test_pivot_table(t_sample_data):
    """
    """

    df = t_sample_data

    pv = pivot_table(df, index=['countries', 'regions'], values='values_1')
    pv.rename(columns={'values_1': 'totals'}, inplace=True)

    expected = 6507.9683290565645
    actual = pv.totals.sum()

    assert expected == actual

# test_pivot_table_sort_ascending_false {{{1
def test_pivot_table_sort_ascending_false(t_sample_data):
    """
    """

    df = t_sample_data

    pv = pivot_table(df, index=['countries'], values='values_1')
    pv.sort_values(by='values_1', ascending=False, inplace=True)

    expected = (8, 1)
    actual = pv.shape

    assert expected == actual

# test_pivot_name_error {{{1
def test_pivot_name_error(t_sample_data):
    """
    Should send log message regarding invalid key/name
    pv object should be None and generate KeyError
    """
    df = t_sample_data

    with pytest.raises(KeyError):
        pv = pivot_table(df, index=['countries_wrong_name'], values='values_1')

# test_pivot_percent_calc {{{1
def test_pivot_percent_calc(t_sample_data):
    """
    """

    df = t_sample_data

    pv = pivot_table(df, index=['countries', 'regions'], values='values_1')
    pv.rename(columns={'values_1': 'totals'}, inplace=True)
    pv.sort_values(by='totals', ascending=False, inplace=True)
    pv['%'] = pv.totals.apply(lambda x: x*100/pv.totals.sum())

    expected = 2.874168873331256
    actual = pv.loc[('Norway','East'), '%']

    assert expected == actual

# test_pivot_cum_percent_calc {{{1
def test_pivot_cum_percent_calc(t_sample_data):
    """
    """
    df = t_sample_data

    pv = pivot_table(df, index=['countries', 'regions'], values='values_1')
    pv.rename(columns={'values_1': 'totals'}, inplace=True)
    pv.sort_values(by='totals', ascending=False, inplace=True)
    pv['%'] = pv.totals.apply(lambda x: x*100/pv.totals.sum())
    pv['cum %'] = pv['%'].cumsum()

    expected = 79.67310369428336
    actual = pv.loc[('Sweden', 'West'), 'cum %']

    assert expected == actual


# test_pivot_table_multi_grouper {{{1
def test_pivot_table_multi_grouper(t_sample_data):
    """
    """
    df = t_sample_data

    p2 = pivot_table(df, index=['dates', 'order_dates',
                                'regions', 'ids'],
                                freq='Q',
                                format_date=True)

    assert p2.loc[('Mar 2020', 'Mar 2020', 'East', 'A'), 'values_1'] > 0


# test_pivot_table_single_grouper {{{1
def test_pivot_table_single_grouper(t_sample_data):
    """
    """
    df = t_sample_data

    p2 = pivot_table(df, index=['dates', 'regions', 'ids'],
                     freq='Q', format_date=True)

    assert p2.loc[('Mar 2020', 'East', 'A'), 'values_1'] > 0


# test_relocate_no_column {{{1
def test_relocate_no_column(t_sample_data):
    """
    """
    df = t_sample_data

    with pytest.raises(KeyError):
        actual = relocate(df, column=None, loc='first')


# test_relocate_index {{{1
def test_relocate_index(t_sample_data):
    """
    """
    df = t_sample_data
    df = df.set_index(['countries', 'regions'])
    df = relocate(df, 'regions', loc='first', index=True)

    expected = ['regions', 'countries']
    actual = df.index.names

    assert expected == actual


# test_relocate_single_column {{{1
def test_relocate_single_column(t_sample_data):
    """
    """
    df = t_sample_data
    df = relocate(df, 'regions', loc='first')

    expected = ['regions', 'dates', 'order_dates', 'countries', 'ids', 'values_1', 'values_2']
    actual = df.columns.values.tolist()
    assert expected == actual


# test_relocate_single_column_last_column {{{1
def test_relocate_single_column_last_column(t_sample_data):
    """
    """
    df = t_sample_data
    df = relocate(df, 'regions', loc='last')

    expected = ['dates', 'order_dates', 'countries', 'ids', 'values_1', 'values_2', 'regions']
    actual = df.columns.values.tolist()
    assert expected == actual


# test_relocate_multi_column_first {{{1
def test_relocate_multi_column_first(t_sample_data):
    """
    """
    df = t_sample_data
    df = relocate(df, ['dates', 'regions', 'countries'], loc='first')

    expected = ['dates', 'regions', 'countries', 'order_dates',
                'ids', 'values_1', 'values_2']

    actual = df.columns.tolist()
    assert expected == actual

# test_relocate_multi_column_last {{{1
def test_relocate_multi_column_last(t_sample_data):
    """
    """
    df = t_sample_data
    df = relocate(df, ['dates', 'regions', 'countries'], loc='last')

    expected = ['order_dates', 'ids', 'values_1', 'values_2',
                'dates', 'regions', 'countries' ]
    actual = df.columns.values.tolist()
    assert expected == actual

# test_relocate_multi_column_before {{{1
def test_relocate_multi_column_before(t_sample_data):
    """
    """
    df = t_sample_data
    df = relocate(df, ['dates', 'regions', 'countries'], loc='before',
                ref_column='values_1')

    expected = ['order_dates', 'ids', 'dates', 'regions',
                'countries', 'values_1', 'values_2']
    actual = df.columns.values.tolist()
    assert expected == actual

# test_relocate_multi_column_after {{{1
def test_relocate_multi_column_after(t_sample_data):
    """
    """
    df = t_sample_data
    df = relocate(df, ['dates', 'regions', 'countries'], loc='after',
                ref_column='order_dates')

    expected = ['order_dates', 'dates', 'regions', 'countries',
                'ids', 'values_1', 'values_2']
    actual = df.columns.values.tolist()
    assert expected == actual


# test_relocate_multi_column_after {{{1
def test_relocate_index_column_after(t_sample_data):
    """
    """
    df = t_sample_data
    df = df.set_index(['countries', 'regions'])
    df = relocate(df, column='countries', loc='after', ref_column='regions')

    expected = 'regions'
    actual = df.index.names[1]

    assert expected == actual


# test_rename {{{1
def test_rename(t_sample_data):
    """
    """

    expected = ['trans_dt', 'order_dates', 'countries',
                'regions', 'ids', 'values_1',
                'values_2']

    df = t_sample_data
    df = rename(df, columns={'dates': 'trans_dt'})
    actual = df.columns.to_list()

    assert expected == actual


# test_rename_axis {{{1
def test_rename_axis(t_sample_data):
    """
    """

    expected = ['AAA', 'BBB']

    df = t_sample_data
    df = pivot_table(df, index=['countries', 'regions'], values='values_1')
    df = rename_axis(df, mapper=('AAA', 'BBB'), axis='rows')
    actual = df.index.names

    assert expected == actual


# test_resample_groupby {{{1
def test_resample_groupby(t_sample_data):
    """ """

    df = t_sample_data

    g1 = group_by(df, by=['dates'], freq='Q').sum()
    g1 = fmt_dateidx(g1, freq='Q')

    # Tidy axis labels
    g1.index.name = 'Date period'
    g1.columns = ['Totals1', 'Totals2']

    assert g1.loc['Mar 2020', 'Totals1'] > 0


# test_resample_multi_grouper_groupby {{{1
def test_resample_multi_grouper_groupby():
    """ """

    prices = {
        'ids': [1, 6, 8, 3],
        'prices': [100, 200, 300, 400],
        'country': ['Canada', 'USA', 'United Kingdom', 'France'],
        'effective': ['2020-01-01', '2020-03-03', '2020-05-30', '2020-10-10'],
        'expired': ['2020-12-31', '2021-04-30', '2022-04-01', '2023-12-31'] }

    df = pd.DataFrame(prices)
    df.effective = pd.to_datetime(df.effective)
    df.expired = pd.to_datetime(df.expired)

    cols = ['country', 'effective', 'expired', 'prices']
    g1 = group_by(df, by=cols, freq='Q').sum()

    expected = (4, 1)
    actual = g1.shape

    assert expected == actual


# test_resample_groupby_multi_index_single_grouper {{{1
def test_resample_groupby_multi_index_single_grouper():
    """
    """
    prices = {
        'ids': [1, 6, 8, 3],
        'prices': [100, 200, 300, 400],
        'country': ['Canada', 'USA', 'United Kingdom', 'France'],
        'effective': ['2020-01-01', '2020-03-03', '2020-05-30', '2020-10-10'],
        'expired': ['2020-12-31', '2021-04-30', '2022-04-01', '2023-12-31'] }

    df = pd.DataFrame(prices)
    df.effective = pd.to_datetime(df.effective)
    df.expired = pd.to_datetime(df.expired)

    cols = ['country', 'effective', 'prices']
    g1 = group_by(df, by=cols, freq='Q').sum()

    expected = (4, 1)
    actual = g1.shape

    assert expected == actual


# test_sample_series {{{1
def test_sample_series(t_simple_series_01):
    """
    """
    s1 = t_simple_series_01

    expected = (2, 1)
    actual = sample(s1, random_state=42).shape

    assert expected == actual


# test_sample_dataframe {{{1
def test_sample_dataframe(t_sample_data):
    """
    """
    df = t_sample_data

    expected = (2, 7)
    actual = sample(df, random_state=42).shape

    assert expected == actual


# test_select_no_parms {{{1
def test_select_no_parms(t_sample_data):

    df = t_sample_data
    df = select(df)

    expected = ['dates', 'order_dates', 'countries',
                'regions', 'ids', 'values_1', 'values_2']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_str_with_regex {{{1
def test_select_str_with_regex(t_sample_data):

    df = t_sample_data
    df = select(df, regex='values')

    expected = ['values_1', 'values_2']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_str_with_regex {{{1
def test_select_str_with_like(t_sample_data):

    df = t_sample_data
    df = select(df, like='values')

    expected = ['values_1', 'values_2']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_slice_with_integers {{{1
def test_select_slice_with_integers(t_sample_data):

    df = t_sample_data
    df = select(df, (3 , 6))

    expected = ['countries', 'regions', 'ids', 'values_1']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_slice_with_column_names {{{1
def test_select_slice_with_column_names(t_sample_data):

    df = t_sample_data
    df = select(df, ('countries', 'values_1'))

    expected = ['countries', 'regions', 'ids', 'values_1']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_column_list {{{1
def test_select_column_list(t_sample_data):

    df = t_sample_data
    df = select(df, ['order_dates'])

    expected = ['order_dates']
    actual = df.columns.tolist()

    assert expected == actual


# test_select_columns_list {{{1
def test_select_columns_list(t_sample_data):

    df = t_sample_data
    df = select(df, ['order_dates', 'countries'])

    expected = ['order_dates', 'countries']
    actual = df.columns.tolist()

    assert expected == actual

# test_select_column_str {{{1
def test_select_column_str(t_sample_data):

    df = t_sample_data
    df = select(df, 'order_dates')

    expected = ['order_dates']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_excluding_column_str {{{1
def test_select_excluding_column_str(t_sample_data):

    df = t_sample_data
    df = select(df, '-order_dates')

    expected = ['dates', 'countries', 'regions',
                'ids', 'values_1', 'values_2']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_excluding_column_list {{{1
def test_select_excluding_column_list(t_sample_data):

    df = t_sample_data
    df = select(df, ['-order_dates'])

    expected = ['dates', 'countries', 'regions',
                'ids', 'values_1', 'values_2']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_columns {{{1
def test_select_columns(t_sample_data):

    df = t_sample_data
    df = select(df, ['dates', 'order_dates'])

    expected = ['dates', 'order_dates']
    actual = df.columns.tolist()

    assert expected == actual


# test_select_include {{{1
def test_select_include(t_sample_data):

    df = t_sample_data
    df = select(df, include='number')

    expected = ['values_1', 'values_2']
    actual = df.columns.tolist()

    assert expected == actual


# test_select_exclude {{{1
def test_select_exclude(t_sample_data):

    df = t_sample_data
    df = select(df, exclude='number')

    expected = ['dates', 'order_dates', 'countries', 'regions', 'ids']
    actual = df.columns.tolist()

    assert expected == actual


# test_select_invalid_column {{{1
def test_select_invalid_column(t_sample_data):

    df = t_sample_data

    with pytest.raises(KeyError):
        actual = select(df, 'AAA')


# test_set_columns {{{1
def test_set_columns():

    data = {'A': ['Order', 'Qty', 10, 40],
            'B': ['Order', 'Number', 12345, 12346]}

    df = pd.DataFrame(data)
    df = set_columns(df, ['C', 'D'])

    expected = ['C', 'D']
    actual = df.columns.to_list()

    assert expected == actual

# test_str_join_str_column_drop_false {{{1
def test_str_join_str_column_drop_false(t_sample_sales):

    df = t_sample_sales

    actual = str_join(df,
                      columns=['actual_sales', 'product'],
                      sep='|',
                      column='combined_col',
                      drop=False)

    assert actual.loc[4, 'combined_col'] == '29209.08|Beachwear'
    assert 'actual_sales' in actual.columns.tolist()
    assert 'product' in actual.columns.tolist()


# test_str_join_str_3_columns_drop_true {{{1
def test_str_join_str_3_columns_drop_true(t_sample_sales):

    df = t_sample_sales

    actual = str_join(df,
                      columns=['actual_sales', 'product', 'actual_profit'],
                      sep='|',
                      column='combined_col',
                      drop=True)

    assert actual.loc[4, 'combined_col'] == '29209.08|Beachwear|1752.54'
    assert 'actual_sales' not in actual.columns.tolist()
    assert 'actual_profit' not in actual.columns.tolist()
    assert 'product' not in actual.columns.tolist()


# test_str_join_str_3_columns_default_join_column_drop_true {{{1
def test_str_join_str_3_columns_default_join_column_drop_true(t_sample_sales):

    df = t_sample_sales

    actual = str_join(df,
                      columns=['actual_sales', 'product', 'actual_profit'],
                      sep='|',
                      drop=True)

    assert actual.loc[4, '0'] == '29209.08|Beachwear|1752.54'
    assert 'actual_sales' not in actual.columns.tolist()
    assert 'actual_profit' not in actual.columns.tolist()
    assert 'product' not in actual.columns.tolist()


# test_str_join_str_column_drop_true {{{1
def test_str_join_str_column_drop_true(t_sample_sales):

    df = t_sample_sales

    actual = str_join(df,
                      columns=['actual_sales', 'product'],
                      sep='|',
                      column='combined_col',
                      drop=True)

    assert actual.loc[4, 'combined_col'] == '29209.08|Beachwear'
    assert 'actual_sales' not in actual.columns.tolist()
    assert 'product' not in actual.columns.tolist()


# test_str_join_str_column_replace_original_column_drop_true {{{1
def test_str_join_str_column_replace_original_column_drop_true(t_sample_sales):

    df = t_sample_sales

    actual = str_join(df,
                      columns=['actual_sales', 'product'],
                      sep='|',
                      column='actual_sales',
                      drop=True)

    assert actual.loc[4, 'actual_sales'] == '29209.08|Beachwear'
    assert 'actual_sales' in actual.columns.tolist()
    assert 'product' not in actual.columns.tolist()


# test_str_split_str_column_drop_false {{{1
def test_str_split_str_column_drop_false(t_sample_sales):

    df = t_sample_sales

    actual = str_split(df, 'product', pat=' ', drop=False)

    # Since columns not specified, default field names provided
    actual[[0, 1, 2]].shape == (200, 3)

    assert actual.shape == (200, 10)


# test_str_split_str_column_drop_true {{{1
def test_str_split_str_column_drop_true(t_sample_sales):

    df = t_sample_sales

    actual = str_split(df, 'product', pat=' ', drop=True)

    # Since columns not specified, default field names provided
    actual[[0, 1, 2]].shape == (200, 3)

    assert actual.shape == (200, 9)


# test_str_split_date_column_drop_true_expand_false {{{1
def test_str_split_date_column_drop_true_expand_false(t_sample_sales):

    df = t_sample_sales

    actual = str_split(df, 'month', pat='-', drop=True, expand=False)

    assert actual.loc[4:4, 'month'].values[0] == ['2021', '01', '01']


# test_str_split_date_column_drop_true_expand_True {{{1
def test_str_split_date_column_drop_true_expand_True(t_sample_sales):

    df = t_sample_sales

    actual = str_split(df, 'month', columns=['year', 'month', 'day'],
                       pat='-', drop=True, expand=True)

    assert actual.loc[4:4, 'year'].values[0] == '2021'


# test_str_split_number_column_drop_true_expand_True {{{1
def test_str_split_number_column_drop_true_expand_True(t_sample_sales):

    df = t_sample_sales

    actual = str_split(df, 'actual_sales',
                       columns=['number', 'precision'],
                       pat='.', drop=True, expand=True)

    assert actual.loc[4:4, 'number'].values[0] == '29209'


# test_summarise_default{{{1
def test_summarise_default(t_sample_data):

    df = t_sample_data

    expected = (7, 2)
    actual = summarise(df)

    assert expected == actual.shape


# test_summarise_value {{{1
def test_summarise_value(t_sample_data):

    df = t_sample_data

    expected = (1, )
    actual = summarise(df, {'values_1': 'sum'})

    assert expected == actual.shape


# test_tail_with_series {{{1
def test_tail_with_series(t_simple_series_01):
    """
    """
    s1 = t_simple_series_01

    expected = (4,)
    actual = tail(s1).shape
    assert expected == actual

# test_tail_with_dataframe {{{1
def test_tail_with_dataframe(t_sample_data):
    """
    """
    df = t_sample_data

    expected = (4, 7)
    actual = tail(df).shape
    assert expected == actual


# test_transform {{{1
def test_transform(t_sample_data):
    """


    """

    index = ['countries', 'regions']
    cols = ['countries', 'regions', 'ids', 'values_1', 'values_2']
    df = t_sample_data[cols]

    gx = transform(df, index=index, g_perc=('values_2', 'percent'))
    gx = transform(df, index=index, total_group_value=('values_2', 'sum'))

    gx.set_index(['countries', 'regions', 'ids'], inplace=True)

    expected = (367, 4)
    actual = gx.shape

    assert expected == actual


# test_transform_with_sort {{{1
def test_transform_with_sort(t_sample_data):
    """
    """

    index = ['countries', 'regions']
    cols = ['countries', 'regions', 'ids', 'values_1', 'values_2']
    df = t_sample_data[cols]

    gx = transform(df, index=index, g_perc=('values_2', 'percent'))
    gx = transform(df, index=index, total_group_value=('values_2', 'sum'))

    gx = gx.set_index(['countries', 'regions', 'ids'])

    expected = (367, 4)
    actual = gx.shape

    assert expected == actual


# test_transform_custom_function {{{1
def test_transform_custom_function(t_sample_data):
    """
    """
    index = ['countries', 'regions']
    cols = ['countries', 'regions', 'ids', 'values_1', 'values_2']
    df = t_sample_data[cols]

    gx = transform(df, index=index, g_perc=('values_2', 'percent'))
    gx = transform(df, index=index, total_group_value=('values_2', lambda x: x.sum()))

    gx.set_index(['countries', 'regions', 'ids'], inplace=True)

    expected = (367, 4)
    actual = gx.shape

    assert expected == actual


# test_str_trim_blanks {{{1
def test_str_trim_blanks(t_sample_column_clean_text):
    """
    """
    df = t_sample_column_clean_text

    df['test_col'] = (df['test_col'].str.replace(r'(\w)\s+(\w)', r'\1 \2', regex=True)
                                    .str.title())

    expected = ['First Row', 'Second Row', 'Fourth Row', 'Fifth Row',
                'Thrid Row', 'Fourth Row', 'Fifth Row', 'Sixth Row',
                'Seventh Row', 'Eighth Row', 'Ninth Row', 'Tenth Row']

    df2 = str_trim(df, str_columns=['test_col'])

    actual = df2['test_col'].to_list()
    assert expected == actual

    str_trim(df, str_columns=None)

    actual = df['test_col'].to_list()
    assert expected == actual

# test_str_trim_blanks_duplicate_column_name {{{1
def test_str_trim_blanks_duplicate_column_name(t_sample_data):
    """
    """
    df = t_sample_data
    df.columns = ['dates', 'order_dates', 'regions', 'regions', 'ids', 'values_1', 'values_2']

    df2 = str_trim(df)

    expected = ['dates', 'order_dates', 'regions', 'regions2', 'ids', 'values_1', 'values_2']
    actual = df2.columns.tolist()
    assert expected == actual

# test_where {{{1
def test_where(t_sample_data):

    df = t_sample_data

    expected = (1, 7)
    actual = where(df, """regions == 'East' and countries == 'Spain' and values_1 == 29""")

    assert expected == actual.shape

# test_inner_join {{{1
def test_inner_join():
    """
    """
    order_data = {'OrderNo': [1001, 1002, 1003, 1004, 1005],
                  'Status': ['A', 'C', 'A', 'A', 'P'],
                  'Type_': ['SO', 'SA', 'SO', 'DA', 'DD']}
    orders = pd.DataFrame(order_data)

    status_data = {'Status': ['A', 'C', 'P'],
                   'description': ['Active', 'Closed', 'Pending']}
    statuses = pd.DataFrame(status_data)

    order_types_data = {'Type_': ['SA', 'SO'],
                        'description': ['Sales Order', 'Standing Order'],
                        'description_2': ['Arbitrary desc', 'another one']}
    types_ = pd.DataFrame(order_types_data)

    merged_df = inner_join(orders, types_, suffixes=('_orders', '_types'))

    expected = (3, 5)
    actual = merged_df.shape

    assert expected == actual


# test_left_join {{{1
def test_left_join():
    """
    """
    order_data = {'OrderNo': [1001, 1002, 1003, 1004, 1005],
                  'Status': ['A', 'C', 'A', 'A', 'P'],
                  'Type_': ['SO', 'SA', 'SO', 'DA', 'DD']}
    orders = pd.DataFrame(order_data)

    status_data = {'Status': ['A', 'C', 'P'],
                   'description': ['Active', 'Closed', 'Pending']}
    statuses = pd.DataFrame(status_data)

    order_types_data = {'Type_': ['SA', 'SO'],
                        'description': ['Sales Order', 'Standing Order'],
                        'description_2': ['Arbitrary desc', 'another one']}
    types_ = pd.DataFrame(order_types_data)

    merged_df = left_join(orders, types_, suffixes=('_orders', '_types'))

    expected = (5, 5)
    actual = merged_df.shape

    assert expected == actual


# test_right_join {{{1
def test_right_join():
    """
    """
    order_data = {'OrderNo': [1001, 1002, 1003, 1004, 1005],
                  'Status': ['A', 'C', 'A', 'A', 'P'],
                  'Type_': ['SO', 'SA', 'SO', 'DA', 'DD']}
    orders = pd.DataFrame(order_data)

    status_data = {'Status': ['A', 'C', 'P'],
                   'description': ['Active', 'Closed', 'Pending']}
    statuses = pd.DataFrame(status_data)

    order_types_data = {'Type_': ['SA', 'SO'],
                        'description': ['Sales Order', 'Standing Order'],
                        'description_2': ['Arbitrary desc', 'another one']}
    types_ = pd.DataFrame(order_types_data)

    merged_df = right_join(orders, types_, suffixes=('_orders', '_types'))

    expected = (3, 5)
    actual = merged_df.shape

    assert expected == actual


# test_outer_join {{{1
def test_outer_join():
    """
    """
    order_data = {'OrderNo': [1001, 1002, 1003, 1004, 1005],
                  'Status': ['A', 'C', 'A', 'A', 'P'],
                  'Type_': ['SO', 'SA', 'SO', 'DA', 'DD']}
    orders = pd.DataFrame(order_data)

    status_data = {'Status': ['A', 'C', 'P'],
                   'description': ['Active', 'Closed', 'Pending']}
    statuses = pd.DataFrame(status_data)

    order_types_data = {'Type_': ['SA', 'SO'],
                        'description': ['Sales Order', 'Standing Order'],
                        'description_2': ['Arbitrary desc', 'another one']}
    types_ = pd.DataFrame(order_types_data)

    merged_df = outer_join(orders, types_, suffixes=('_orders', '_types'))

    expected = (5, 5)
    actual = merged_df.shape

    assert expected == actual
