from piper.pandas import read_csv
from piper.verbs import add_formula
from piper.verbs import adorn
from piper.verbs import assign
from piper.verbs import clean_columns
from piper.verbs import columns
from piper.verbs import combine_header_rows
from piper.verbs import count
from piper.verbs import distinct
from piper.verbs import drop
from piper.verbs import duplicated
from piper.verbs import explode
from piper.verbs import flatten_cols
from piper.verbs import format_dateindex
from piper.verbs import group_by
from piper.verbs import transform
from piper.verbs import has_special_chars
from piper.verbs import head
from piper.verbs import info
from piper.verbs import inner_join
from piper.verbs import left_join
from piper.verbs import outer_join
from piper.verbs import overlaps
from piper.verbs import order_by
from piper.verbs import pivot_table
from piper.verbs import relocate
from piper.verbs import rename
from piper.verbs import rename_axis
from piper.verbs import right_join
from piper.verbs import sample
from piper.verbs import select
from piper.verbs import set_columns
from piper.verbs import summarise
from piper.verbs import tail
from piper.verbs import to_tsv
from piper.verbs import trim
from piper.verbs import where
from piper.test.factory import get_sample_orders_01
from piper.test.factory import get_sample_df1
from piper.test.factory import get_sample_df2
from piper.test.factory import get_sample_df3
from piper.test.factory import get_sample_df4
from piper.test.factory import get_sample_df5
from piper.test.factory import get_sample_df6
from piper.test.factory import get_sample_df7
from pandas._testing import assert_frame_equal
from pandas._testing import assert_series_equal
import random
import json
import numpy as np
import pandas as pd
import pytest

# sample_orders_01 {{{1
@pytest.fixture
def sample_orders_01():
    return get_sample_orders_01()

# sample_df1 {{{1
@pytest.fixture
def sample_df1():
    return get_sample_df1()

# sample_df2 {{{1
@pytest.fixture
def sample_df2():
    return get_sample_df2()

# sample_df3 {{{1
@pytest.fixture
def sample_df3():
    return get_sample_df3()

# sample_df4 {{{1
@pytest.fixture
def sample_df4():
    return get_sample_df4()

# sample_df5 {{{1
@pytest.fixture
def sample_df5():
    return get_sample_df5()

# sample_df6 {{{1
@pytest.fixture
def sample_df6():
    return get_sample_df6()

# sample_df7 {{{1
@pytest.fixture
def sample_df7():
    return get_sample_df7()


# sample_s1 {{{1
@pytest.fixture
def sample_s1():

    id_list = ['A', 'B', 'C', 'D', 'E']
    s1 = pd.Series(np.random.choice(id_list, size=5), name='ids')

    return s1


# get_column_list {{{1
@pytest.fixture
def get_column_list():
    """ Column list for dataframe
    """
    column_list = ['dupe', 'Customer', 'Target name', 'Public',
                   '_  Material', 'Prod type', 'Effective     ',
                   'Expired', 'Price  ', 'Currency']

    return column_list

# test_head_with_series {{{1
def test_head_with_series(sample_s1):
    """
    """
    s1 = sample_s1

    expected = (4,)
    actual = head(s1).shape
    assert expected == actual

# test_head_with_dataframe {{{1
def test_head_with_dataframe(sample_df2):
    """
    """
    df = sample_df2

    expected = (4, 2)
    actual = head(df).shape
    assert expected == actual

# test_head_with_columns_function {{{1
def test_head_with_columns_function(sample_df1):
    """
    """
    df = sample_df1

    expected = (4, 4)

    actual = head(df[columns(df, regex='order|dates|values',
        astype='list')]).shape

    assert expected == actual

# test_tail_with_series {{{1
def test_tail_with_series(sample_s1):
    """
    """
    s1 = sample_s1

    expected = (4,)
    actual = tail(s1).shape
    assert expected == actual

# test_tail_with_dataframe {{{1
def test_tail_with_dataframe(sample_df2):
    """
    """
    df = sample_df2

    expected = (4, 2)
    actual = tail(df).shape
    assert expected == actual



# test_assign {{{1
def test_assign(sample_df1):
    """
    """
    df = sample_df1
    df = where(df, "ids == 'A'")
    df = where(df, "values_1 > 300 & countries.isin(['Italy', 'Spain'])")
    df = assign(df, new_field=lambda x: x.countries.str[:3]+x.regions,
                       another=lambda x:3*x.values_1)

    expected = ['dates', 'order_dates', 'countries',
                'regions', 'ids', 'values_1',
                'values_2', 'new_field', 'another']
    actual = df.columns.tolist()

    assert expected == actual


# test_drop {{{1
def test_drop(sample_df1):
    """
    """
    df = sample_df1
    df = drop(df, columns=['countries', 'regions'])

    expected = ['dates', 'order_dates', 'ids', 'values_1', 'values_2']
    actual = df.columns.tolist()

    assert expected == actual


# test_relocate_index {{{1
def test_relocate_index(sample_df1):
    """
    """
    df = sample_df1
    df = df.set_index(['countries', 'regions'])
    df = relocate(df, 'regions', loc='first', index=True)

    expected = ['regions', 'countries']
    actual = df.index.names

    assert expected == actual


# test_relocate_single_column {{{1
def test_relocate_single_column(sample_df2):
    """
    """
    df = sample_df2
    df = relocate(df, 'regions', loc='first')

    expected = ['regions', 'ids']
    actual = df.columns.values.tolist()
    assert expected == actual


# test_relocate_single_column_last_column {{{1
def test_relocate_single_column_last_column(sample_df2):
    """
    """
    df = sample_df2
    df = relocate(df, 'regions', loc='last')

    expected = ['ids', 'regions']
    actual = df.columns.values.tolist()
    assert expected == actual


# test_relocate_multi_column_first {{{1
def test_relocate_multi_column_first(sample_df1):
    """
    """
    df = sample_df1
    df = relocate(df, ['dates', 'regions', 'countries'], loc='first')

    expected = ['dates', 'regions', 'countries', 'order_dates',
                'ids', 'values_1', 'values_2']

    actual = df.columns.tolist()
    assert expected == actual

# test_relocate_multi_column_last {{{1
def test_relocate_multi_column_last(sample_df1):
    """
    """
    df = sample_df1
    df = relocate(df, ['dates', 'regions', 'countries'], loc='last')

    expected = ['order_dates', 'ids', 'values_1', 'values_2',
                'dates', 'regions', 'countries' ]
    actual = df.columns.values.tolist()
    assert expected == actual

# test_relocate_multi_column_before {{{1
def test_relocate_multi_column_before(sample_df1):
    """
    """
    df = sample_df1
    df = relocate(df, ['dates', 'regions', 'countries'], loc='before',
                ref_column='values_1')

    expected = ['order_dates', 'ids', 'dates', 'regions',
                'countries', 'values_1', 'values_2']
    actual = df.columns.values.tolist()
    assert expected == actual

# test_relocate_multi_column_after {{{1
def test_relocate_multi_column_after(sample_df1):
    """
    """
    df = sample_df1
    df = relocate(df, ['dates', 'regions', 'countries'], loc='after',
                ref_column='order_dates')

    expected = ['order_dates', 'dates', 'regions', 'countries',
                'ids', 'values_1', 'values_2']
    actual = df.columns.values.tolist()
    assert expected == actual


# test_relocate_multi_column_after {{{1
def test_relocate_index_column_after(sample_df1):
    """
    """
    df = sample_df1
    df = df.set_index(['countries', 'regions'])
    df = relocate(df, column='countries', loc='after', ref_column='regions')

    expected = 'regions'
    actual = df.index.names[1]

    assert expected == actual


# test_flatten_cols_no_index {{{1
def test_flatten_cols_no_index(sample_df1):
    """
    """
    df = sample_df1
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
def test_flatten_cols_keep_prefix(sample_df1):
    """
    """
    df = sample_df1
    df = group_by(df, ['countries', 'regions'])
    df = summarise(df, total=('values_2', 'sum'))
    df = df.unstack()
    df = flatten_cols(df)

    expected = ['total_East', 'total_North', 'total_South', 'total_West']
    actual = df.columns.to_list()

    assert expected == actual


# test_flatten_cols_lose_prefix {{{1
def test_flatten_cols_lose_prefix(sample_df1):
    """
    """
    df = sample_df1
    df = group_by(df, ['countries', 'regions'])
    df = summarise(df, total=('values_2', 'sum'))
    df = df.unstack()
    df = flatten_cols(df, remove_prefix='total')

    expected = ['East', 'North', 'South', 'West']
    actual = df.columns.to_list()

    assert expected == actual


# test_flatten_cols_remove_prefix {{{1
def test_flatten_cols_remove_prefix(sample_df1):
    """
    """
    df = sample_df1
    df = group_by(df, ['countries', 'regions'])
    df = summarise(df, total=('values_2', 'sum'))
    df = df.unstack()
    df = flatten_cols(df, remove_prefix='total')

    expected = ['East', 'North', 'South', 'West']
    actual = df.columns.to_list()

    assert expected == actual


# test_combine_header_rows {{{1
def test_combine_header_rows():

    data = {'A': ['Order', 'Qty', 10, 40],
            'B': ['Order', 'Number', 12345, 12346]}

    df = pd.DataFrame(data)
    df = combine_header_rows(df)

    expected = ['Order Qty', 'Order Number']
    actual = df.columns.to_list()

    assert expected == actual


# test_combine_header_rows_lower {{{1
def test_combine_header_rows_lower():

    data = {'A': ['Order', 'Qty', 10, 40],
            'B': ['Order', 'Number', 12345, 12346]}

    df = pd.DataFrame(data)
    df = combine_header_rows(df, apply='lower')

    expected = ['order qty', 'order number']
    actual = df.columns.to_list()

    assert expected == actual


# test_combine_header_rows_title {{{1
def test_combine_header_rows_title():

    data = {'A': ['Order', 'Qty', 10, 40],
            'B': ['Order', 'Number', 12345, 12346]}

    df = pd.DataFrame(data)
    df = combine_header_rows(df, apply='title')

    expected = ['Order Qty', 'Order Number']
    actual = df.columns.to_list()

    assert expected == actual


# test_set_columns {{{1
def test_set_columns():

    data = {'A': ['Order', 'Qty', 10, 40],
            'B': ['Order', 'Number', 12345, 12346]}

    df = pd.DataFrame(data)
    df = set_columns(df, ['C', 'D'])

    expected = ['C', 'D']
    actual = df.columns.to_list()

    assert expected == actual


# test_clean_column_title {{{1
def test_clean_column_list_using_list_and_title(get_column_list):
    """
    """
    expected = ['Dupe', 'Customer', 'Target_Name', 'Public',
                'Material', 'Prod_Type', 'Effective', 'Expired',
                'Price', 'Currency']

    dx = pd.DataFrame(None, columns=get_column_list)

    actual = clean_columns(dx, title=True).columns.tolist()
    assert expected == actual

# test_clean_column_list_using_df_columns {{{1
def test_clean_column_list_using_df_columns(get_column_list):
    """
    """
    expected = ['dupe', 'customer', 'target_name', 'public',
                'material', 'prod_type', 'effective', 'expired',
                'price', 'currency']

    dx = pd.DataFrame(None, columns=get_column_list)

    actual = clean_columns(dx).columns.tolist()
    assert expected == actual

# test_trim_blanks {{{1
def test_trim_blanks(sample_df7):
    """
    """
    df = sample_df7

    df['test_col'] = (df['test_col'].str.replace(r'(\w)\s+(\w)', r'\1 \2', regex=True)
                                    .str.title())

    expected = ['First Row', 'Second Row', 'Fourth Row', 'Fifth Row',
                'Thrid Row', 'Fourth Row', 'Fifth Row', 'Sixth Row',
                'Seventh Row', 'Eighth Row', 'Ninth Row', 'Tenth Row']

    df2 = trim(df, str_columns=['test_col'], inplace=False)

    actual = df2['test_col'].to_list()
    assert expected == actual

    trim(df, str_columns=None, inplace=True)

    actual = df['test_col'].to_list()
    assert expected == actual

# test_trim_blanks_duplicate_column_name {{{1
def test_trim_blanks_duplicate_column_name(sample_df1):
    """
    """
    df = sample_df1
    df.columns = ['dates', 'order_dates', 'regions', 'regions', 'ids', 'values_1', 'values_2']

    df2 = trim(df)

    expected = ['dates', 'order_dates', 'regions', 'regions2', 'ids', 'values_1', 'values_2']
    actual = df2.columns.tolist()
    assert expected == actual

# test_columns_as_list {{{1
def test_columns_as_list(sample_df2):

    df = sample_df2

    expected = ['ids', 'regions']
    actual = columns(df, astype='list')

    assert expected == actual

# test_columns_as_dict {{{1
def test_columns_as_dict(sample_df2):

    df = sample_df2

    expected = {'ids': 'ids', 'regions': 'regions'}
    actual = columns(df, astype='dict')

    assert expected == actual

# test_columns_as_dataframe {{{1
def test_columns_as_dataframe(sample_df2):

    df = sample_df2

    expected = pd.DataFrame(['ids', 'regions'], columns=['column_names'])
    actual = columns(df, astype='dataframe')

    assert_frame_equal(expected, actual)

# test_columns_as_series {{{1
def test_columns_as_series(sample_df2):

    df = sample_df2

    expected = pd.Series(['ids', 'regions'], index=range(2), name='column_names')
    actual = columns(df, astype='series')

    assert_series_equal(expected, actual)

# test_columns_as_text {{{1
def test_columns_as_text(sample_df2):

    df = sample_df2

    expected = "['ids', 'regions']"
    actual = columns(df, astype='text')

    assert expected == actual

# test_columns_as_regex {{{1
def test_columns_as_regex(sample_df1):

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

# test_distinct {{{1
def test_distinct(sample_df1):

    df = sample_df1
    df = select(df, ['countries', 'regions', 'ids'])
    df = distinct(df, 'ids')

    expected = (5, 3)
    actual = df.shape

    assert expected == actual


# test_select_no_parms {{{1
def test_select_no_parms(sample_df1):

    df = sample_df1
    df = select(df)

    expected = ['dates', 'order_dates', 'countries',
                'regions', 'ids', 'values_1', 'values_2']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_str_with_regex {{{1
def test_select_str_with_regex(sample_df1):

    df = sample_df1
    df = select(df, 'values', regex=True)

    expected = ['values_1', 'values_2']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_slice {{{1
def test_select_slice(sample_df1):

    df = sample_df1
    df = select(df, slice('countries', 'values_1'))

    expected = ['countries', 'regions', 'ids', 'values_1']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_column_list {{{1
def test_select_column_list(sample_df1):

    df = sample_df1
    df = select(df, ['order_dates'])

    expected = ['order_dates']
    actual = df.columns.tolist()

    assert expected == actual


# test_select_columns_list {{{1
def test_select_columns_list(sample_df1):

    df = sample_df1
    df = select(df, ['order_dates', 'countries'])

    expected = ['order_dates', 'countries']
    actual = df.columns.tolist()

    assert expected == actual

# test_select_column_str {{{1
def test_select_column_str(sample_df1):

    df = sample_df1
    df = select(df, 'order_dates')

    expected = ['order_dates']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_excluding_column_str {{{1
def test_select_excluding_column_str(sample_df1):

    df = sample_df1
    df = select(df, '-order_dates')

    expected = ['dates', 'countries', 'regions',
                'ids', 'values_1', 'values_2']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_excluding_column_list {{{1
def test_select_excluding_column_list(sample_df1):

    df = sample_df1
    df = select(df, ['-order_dates'])

    expected = ['dates', 'countries', 'regions',
                'ids', 'values_1', 'values_2']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_columns {{{1
def test_select_columns(sample_df1):

    df = sample_df1
    df = select(df, ['dates', 'order_dates'])

    expected = ['dates', 'order_dates']
    actual = df.columns.tolist()

    assert expected == actual


# test_select_invalid_column {{{1
def test_select_invalid_column(sample_df1):

    df = sample_df1

    expected = None
    actual = select(df, 'AAA')

    assert expected == actual


# test_info {{{1
def test_info(sample_df5):

    df = sample_df5

    expected = (2, 6)
    actual = info(df).shape

    assert expected == actual

# test_info_with_dupes {{{1
def test_info_with_dupes(sample_df5):

    df = sample_df5

    expected = (2, 7)
    actual = info(df, n_dupes=True).shape

    assert expected == actual

# test_info_with_na_cols {{{1
def test_info_with_na_cols(sample_df5):

    df = sample_df5

    expected = (0, 6)
    actual = info(df, fillna=True).shape

    assert expected == actual

# test_count {{{1
def test_count(sample_df5):

    df = sample_df5

    expected = (1, 3)
    actual = count(df, 'bgy56icnt').shape

    assert expected == actual


# test_count_series {{{1
def test_count_series(sample_df5):

    df = sample_df5

    expected = (1, 3)
    actual = count(df.bgy56icnt).shape

    assert expected == actual

# test_count_sort {{{1
def test_count_sort(sample_df5):

    df = sample_df5

    expected = (1, 3)
    actual = count(df, 'bgy56icnt', sort_values=True).shape

    assert expected == actual

# test_count_with_total {{{1
def test_count_with_total(sample_df5):

    df = sample_df5

    expected = (2, 3)
    actual = count(df, 'bgy56icnt', totals=True, sort_values=True).shape

    assert expected == actual


# test_count_with_total_percent_cum_percent {{{1
def test_count_with_total_percent_cum_percent(sample_df5):

    df = sample_df5

    expected = (2, 3)
    actual = count(df, 'bgy56icnt', totals=True, sort_values=True,
                   percent=True, cum_percent=True).shape

    assert expected == actual


# test_count_with_cum_percent_with_threshold {{{1
def test_count_with_cum_percent_with_threshold(sample_df1):

    expected = (6, 3)

    df = count(sample_df1, 'countries', threshold = 80, cum_percent=True)
    actual = df.shape

    assert expected == actual


# test_count_not_found_column {{{1
def test_count_not_found_column(sample_df1):

    df = sample_df1

    expected = None
    actual = count(df, 'invalid_column')

    assert expected == actual


# test_count_no_column {{{1
def test_count_no_column(sample_df1):

    df = sample_df1

    expected = (7, 3)
    actual = count(df).shape

    assert expected == actual


# test_count_column_reset_index_true {{{1
def test_count_column_reset_index_true(sample_df1):

    df = sample_df1

    expected = (8, 4)
    actual = count(df, 'countries', reset_index=True).shape

    assert expected == actual


# test_count_single_column {{{1
def test_count_single_column(sample_df1):

    df = sample_df1

    expected = (4, 3)
    actual = count(df, 'regions').shape

    assert expected == actual


# test_count_multi_column {{{1
def test_count_multi_column(sample_df1):

    df = sample_df1

    query = "regions == 'East' and countries.isin(['Italy'])"
    df = df.query(query)

    expected = (1, 5)
    actual = count(df, ['regions', 'countries']).reset_index().shape

    assert expected == actual


# test_count_multi_column_with_percent {{{1
def test_count_multi_column_with_percent(sample_df1):

    df = sample_df1
    df = df.query("regions == 'East' and countries.isin(['Italy'])")
    df = count(df, ['regions', 'countries'], percent=True, cum_percent=True)

    expected = 100.0
    actual = df.loc[('East', 'Italy'), 'cum %']

    assert expected == actual


# test_count_multi_column_with_cum_percent_threshold {{{1
def test_count_multi_column_with_cum_percent_threshold(sample_df1):

    df = sample_df1
    df = df.query("regions == 'East'")
    df = count(df, ['regions', 'countries'],
               percent=True,
               cum_percent=True,
               threshold=81)

    expected = (5, 3)
    actual = df.shape

    assert expected == actual


# test_where {{{1
def test_where(sample_df1):

    df = sample_df1

    expected = (1, 7)
    actual = where(df, """regions == 'East' and countries == 'Spain' and values_1 == 29""")

    assert expected == actual.shape

# test_summarise_default{{{1
def test_summarise_default(sample_df1):

    df = sample_df1

    expected = (7, )
    actual = summarise(df)

    assert expected == actual.shape


# test_summarise_value {{{1
def test_summarise_value(sample_df1):

    df = sample_df1

    expected = (1, )
    actual = summarise(df, {'values_1': 'sum'})

    assert expected == actual.shape


# test_add_formula {{{1
def test_add_formula(sample_df2):

    df = sample_df2

    formula = '=CONCATENATE(A{row}, B{row}, C{row})'
    add_formula(df, column_name='X7', loc=2, formula=formula, inplace=True)

    expected = (5, )

    assert expected == df.X7.shape

# test_add_formula_pos_last {{{1
def test_add_formula_pos_last(sample_df2):

    df = sample_df2

    formula = '=CONCATENATE(A{row}, B{row}, C{row})'
    add_formula(df, column_name='X7', loc='last',
                formula=formula, inplace=True)

    assert df.columns[-1] == 'X7'


# test_add_formula_inplace_false {{{1
def test_add_formula_inplace_false(sample_df2):

    df = sample_df2

    expected = df

    formula = '=CONCATENATE(A{row}, B{row}, C{row})'

    actual = add_formula(df, column_name='X7', loc=2,
                         formula=formula, inplace=False)


    assert expected.shape != actual.shape


# test_overlaps {{{1
def test_overlaps():

    prices = {
        'prices': [100, 200, 300],
        'contract': ['A', 'B', 'A'],
        'effective': ['2020-01-01', '2020-03-03', '2020-05-30'],
        'expired': ['2020-12-31', '2021-04-30', '2022-04-01']
    }

    df = pd.DataFrame(prices)

    expected = (2, 6)
    actual = overlaps(df, effective='effective', expired='expired',
                      price='prices', unique_cols=['contract'])

    assert expected == actual.shape


# test_overlaps_pos_first {{{1
def test_overlaps_pos_first():

    prices = {
        'prices': [100, 200, 300],
        'contract': ['A', 'B', 'A'],
        'effective': ['2020-01-01', '2020-03-03', '2020-05-30'],
        'expired': ['2020-12-31', '2021-04-30', '2022-04-01']
    }

    df = pd.DataFrame(prices)

    expected = 'price_diff'
    actual = overlaps(df, effective='effective', loc='first',
                      expired='expired', price='prices',
                      unique_cols=['contract'])

    assert expected == actual.columns.tolist()[0]

# test_overlaps_raises_key_error {{{1
def test_overlaps_raises_key_error():

    prices = {
        'prices': [100, 200, 300],
        'contract': ['A', 'B', 'A'],
        'effective': ['2020-01-01', '2020-03-03', '2020-05-30'],
        'expired': ['2020-12-31', '2021-04-30', '2022-04-01']
    }

    df = pd.DataFrame(prices)

    with pytest.raises(KeyError):
        actual = overlaps(df, effective='false_field', expired='expired',
                          price='prices', unique_cols=['contract'])

# test_overlaps_no_price {{{1
def test_overlaps_no_price():

    prices = {
        'prices': [100, 200, 300],
        'contract': ['A', 'B', 'A'],
        'effective': ['2020-01-01', '2020-03-03', '2020-05-30'],
        'expired': ['2020-12-31', '2021-04-30', '2022-04-01']
    }

    df = pd.DataFrame(prices)

    expected = (2, 5)
    actual = overlaps(df, effective='effective', expired='expired',
                      price=False, unique_cols=['contract'])

    assert expected == actual.shape


# test_inner_join {{{1
def test_inner_join(sample_df6):
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
def test_left_join(sample_df6):
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
def test_right_join(sample_df6):
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
def test_outer_join(sample_df6):
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


# test_sample_series {{{1
def test_sample_series(sample_s1):
    """
    """
    df = sample_s1

    expected = (2, 1)
    actual = sample(df, random_state=42).shape

    assert expected == actual


# test_explode {{{1
def test_explode(sample_df1):
    """
    """
    df = sample_df1
    df = group_by(df, 'countries')
    df = summarise(df, ids=('ids', set))

    expected = (40, 1)
    actual = explode(df, 'ids').shape

    assert expected == actual


# test_rename {{{1
def test_rename(sample_df1):
    """
    """

    expected = ['trans_dt', 'order_dates', 'countries',
                'regions', 'ids', 'values_1',
                'values_2']

    df = sample_df1
    df = rename(df, columns={'dates': 'trans_dt'})
    actual = df.columns.to_list()

    assert expected == actual


# test_rename_axis {{{1
def test_rename_axis(sample_df1):
    """
    """

    expected = ['AAA', 'BBB']

    df = sample_df1
    df = pivot_table(df, index=['countries', 'regions'], values='values_1')
    df = rename_axis(df, mapper=('AAA', 'BBB'), axis='rows')
    actual = df.index.names

    assert expected == actual


# test_sample_dataframe {{{1
def test_sample_dataframe(sample_df1):
    """
    """
    df = sample_df1

    expected = (2, 7)
    actual = sample(df, random_state=42).shape

    assert expected == actual


# test_resample_groupby {{{1
def test_resample_groupby(sample_df1):
    """ """

    df = sample_df1

    g1 = group_by(df, by=['dates'], freq='Q').sum()
    g1 = format_dateindex(g1, freq='Q')

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


# test_pivot_table {{{1
def test_pivot_table(sample_df1):
    """
    """

    df = sample_df1

    pv = pivot_table(df, index=['countries', 'regions'], values='values_1')
    pv.rename(columns={'values_1': 'totals'}, inplace=True)

    expected = 6507.9683290565645
    actual = pv.totals.sum()

    assert expected == actual

# test_pivot_table_sort_ascending_false {{{1
def test_pivot_table_sort_ascending_false(sample_df1):
    """
    """

    df = sample_df1

    pv = pivot_table(df, index=['countries'], values='values_1')
    pv.sort_values(by='values_1', ascending=False, inplace=True)

    expected = (8, 1)
    actual = pv.shape

    assert expected == actual

# test_pivot_name_error {{{1
def test_pivot_name_error(sample_df1):
    """
    Should send log message regarding invalid key/name
    pv object should be None and generate KeyError
    """
    df = sample_df1

    with pytest.raises(KeyError):
        pv = pivot_table(df, index=['countries_wrong_name'], values='values_1')
        pv.sort_values(by='values_1', ascending=False, inplace=True)
        pv.shape

# test_pivot_percent_calc {{{1
def test_pivot_percent_calc(sample_df1):
    """
    """

    df = sample_df1

    pv = pivot_table(df, index=['countries', 'regions'], values='values_1')
    pv.rename(columns={'values_1': 'totals'}, inplace=True)
    pv.sort_values(by='totals', ascending=False, inplace=True)
    pv['%'] = pv.totals.apply(lambda x: x*100/pv.totals.sum())

    expected = 2.874168873331256
    actual = pv.loc[('Norway','East'), '%']

    assert expected == actual

# test_pivot_cum_percent_calc {{{1
def test_pivot_cum_percent_calc(sample_df1):
    """
    """
    df = sample_df1

    pv = pivot_table(df, index=['countries', 'regions'], values='values_1')
    pv.rename(columns={'values_1': 'totals'}, inplace=True)
    pv.sort_values(by='totals', ascending=False, inplace=True)
    pv['%'] = pv.totals.apply(lambda x: x*100/pv.totals.sum())
    pv['cum %'] = pv['%'].cumsum()

    expected = 79.67310369428336
    actual = pv.loc[('Sweden', 'West'), 'cum %']

    assert expected == actual


# test_pivot_table_multi_grouper {{{1
def test_pivot_table_multi_grouper(sample_df1):
    """
    """
    df = sample_df1

    p2 = pivot_table(df, index=['dates', 'order_dates',
                                'regions', 'ids'],
                                freq='Q',
                                format_date=True)

    assert p2.loc[('Mar 2020', 'Mar 2020', 'East', 'A'), 'values_1'] > 0


# test_pivot_table_single_grouper {{{1
def test_pivot_table_single_grouper(sample_df1):
    """
    """
    df = sample_df1

    p2 = pivot_table(df, index=['dates', 'regions', 'ids'],
                     freq='Q', format_date=True)

    assert p2.loc[('Mar 2020', 'East', 'A'), 'values_1'] > 0


# test_transform {{{1
def test_transform(sample_df1):
    """
    """

    index = ['countries', 'regions']
    cols = ['countries', 'regions', 'ids', 'values_1', 'values_2']
    df = sample_df1[cols]

    gx = transform(df, index=index, g_perc=('values_2', 'percent'))
    gx = transform(df, index=index, total_group_value=('values_2', 'sum'))

    gx.set_index(['countries', 'regions', 'ids'], inplace=True)

    expected = (367, 4)
    actual = gx.shape

    assert expected == actual


# test_transform_with_sort {{{1
def test_transform_with_sort(sample_df1):
    """
    """

    index = ['countries', 'regions']
    cols = ['countries', 'regions', 'ids', 'values_1', 'values_2']
    df = sample_df1[cols]

    gx = transform(df, index=index, g_perc=('values_2', 'percent'))
    gx = transform(df, index=index, total_group_value=('values_2', 'sum'))

    gx = gx.set_index(['countries', 'regions', 'ids'])

    expected = (367, 4)
    actual = gx.shape

    assert expected == actual


# test_transform_custom_function {{{1
def test_transform_custom_function(sample_df1):
    """
    """
    index = ['countries', 'regions']
    cols = ['countries', 'regions', 'ids', 'values_1', 'values_2']
    df = sample_df1[cols]

    gx = transform(df, index=index, g_perc=('values_2', 'percent'))
    gx = transform(df, index=index, total_group_value=('values_2', lambda x: x.sum()))

    gx.set_index(['countries', 'regions', 'ids'], inplace=True)

    expected = (367, 4)
    actual = gx.shape

    assert expected == actual


# test_adorn_row_total {{{1
def test_adorn_row_total(sample_df1):

    df = sample_df1
    df = group_by(df, ['countries'])
    df = summarise(df, total=('values_1', 'sum'))
    df = adorn(df)

    expected = 73604
    actual = df.loc['All'].values[0]

    assert expected == actual


# test_adorn_column_total {{{1
def test_adorn_column_total(sample_df1):

    df = sample_df1
    df = group_by(df, ['countries'])
    df = summarise(df, total=('values_1', 'sum'))
    df = adorn(df, axis = 'column')

    expected = 8432
    actual = df.loc['Sweden', 'All']

    assert expected == actual


# test_adorn_with_ignore_row_index {{{1
def test_adorn_row_with_ignore_row_index(sample_df1):

    df = sample_df1
    df = group_by(df, ['countries'])
    df = summarise(df, total=('values_1', 'sum'))
    df = adorn(df, axis = 'row', ignore_row_index=True)

    expected = 'All'
    actual = df.iloc[df.shape[0]-1, 0]

    assert expected == actual


# test_adorn_column_with_column_specified {{{1
def test_adorn_column_with_column_specified(sample_df1):

    df = sample_df1
    df = group_by(df, ['countries'])
    df = summarise(df, total=('values_1', 'sum'))
    df = adorn(df, columns='total', axis = 'column')

    expected = 8432
    actual = df.loc['Sweden', 'All']

    assert expected == actual

# test_adorn_column_with_column_list_specified {{{1
def test_adorn_column_with_column_list_specified(sample_df1):

    df = sample_df1
    df = group_by(df, ['countries', 'regions'])
    df = summarise(df, total=('values_1', 'sum'))
    df = assign(df, total2=lambda x: x.total * 10)
    df = adorn(df, columns=['total', 'total2'], axis = 'both')

    expected = ['total', 'total2', 'All']
    actual = df.columns.tolist()

    assert expected == actual

# test_adorn_column_with_column_str_specified {{{1
def test_adorn_column_with_column_str_specified(sample_df1):

    df = sample_df1
    df = group_by(df, ['countries', 'regions'])
    df = summarise(df, total=('values_1', 'sum'))
    df = assign(df, total2=lambda x: x.total * 10)
    df = adorn(df, columns='total', axis = 'both')

    expected = ['total', 'total2', 'All']
    actual = df.columns.tolist()

    assert expected == actual


# test_has_special_chars {{{1
def test_has_special_chars(sample_df1):
    """
    """
    def generate(random_chars=10):
        alphabet="0123456789abcdefghijklmnopqrstuvwxyz!$%()"

        r = random.SystemRandom()
        chars = ''.join([r.choice(alphabet) for i in range(random_chars)])

        return chars

    dx = pd.DataFrame([generate() for x in range(30)], columns=['col'])

    df_count = count(has_special_chars(dx, 'col'), 'item_special_chars')

    assert df_count.shape == (2, 3)


# test_duplicated {{{1
def test_duplicated(sample_df5):
    """
    """
    df = sample_df5

    duplicated_fields = ['bgy56icnt', 'bgz56ccode']
    df2 = duplicated(df[duplicated_fields],
                     subset=duplicated_fields,
                     keep='first',
                     sort=True,
                     column='dupe',
                     loc='first')

    expected = 2  # Duplicate records
    actual = df2.dupe.value_counts()[1]
    assert expected == actual

# test_duplicated_first_position {{{1
def test_duplicated_first_position(sample_df5):
    """
    """
    df = sample_df5

    duplicated_fields = ['bgy56icnt', 'bgz56ccode']
    df2 = duplicated(df[duplicated_fields],
                     subset=duplicated_fields,
                     keep='first',
                     sort=True,
                     column='dupe',
                     loc='first')

    expected = 'dupe'
    actual = df2.columns.tolist()[0]
    assert expected == actual

# test_duplicated_last_position {{{1
def test_duplicated_last_position(sample_df5):
    """
    """
    df = sample_df5

    duplicated_fields = ['bgy56icnt', 'bgz56ccode']
    df2 = duplicated(df[duplicated_fields],
                     subset=duplicated_fields,
                     keep='first',
                     sort=True,
                     column='dupe',
                     loc='last')

    expected = 'dupe'
    actual = df2.columns.tolist()[-1]
    assert expected == actual

# test_duplicated_inplace_true {{{1
def test_duplicated_inplace_true(sample_df5):
    """
    """
    df = sample_df5

    duplicated_fields = ['bgy56icnt', 'bgz56ccode']

    duplicated(df[duplicated_fields],
            subset=duplicated_fields,
            keep='first', sort=True,
            column='dupe',
            loc='first')

    assert_frame_equal(df, df.copy(deep=True))

# test_duplicated_data_raise_col_position_error {{{1
def test_duplicated_data_raise_col_position_error(sample_df5):
    """
    """
    df = sample_df5
    duplicated_fields = ['bgy56icnt', 'bgz56ccode']

    with pytest.raises(Exception):
        assert duplicated(df[duplicated_fields],
                          subset=duplicated_fields, keep='first',
                          sort=True, insert_column=True,
                          column_name='dupe', column_pos=6) is None

# test_to_tsv_with_data {{{1
def test_to_tsv_with_data(sample_orders_01):
    """
    Write sample dataframe, no value is return from to_tsv
    """
    file_name = 'piper/temp/to_tsv_with_data.tsv'
    df = pd.DataFrame(sample_orders_01)

    expected = None
    actual = to_tsv(df, file_name, sep='\t')

    assert expected == actual


# test_read_csv_with_data {{{1
def test_read_csv_with_data(sample_orders_01):
    """
    """
    file_name = 'piper/temp/to_tsv_with_data.tsv'
    df_json = pd.DataFrame(sample_orders_01)

    df = read_csv(file_name, sep='\t')

    expected = df_json.shape
    actual = df.shape

    assert expected == actual
