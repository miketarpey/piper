from piper.verbs import to_tsv
from piper.pandas import read_csv
from piper.verbs import left_join
from piper.verbs import duplicated
from piper.verbs import resample_groupby
from piper.verbs import resample_pivot
from piper.verbs import add_group_calc
from piper.verbs import overlaps
from piper.verbs import add_formula
from piper.verbs import clean_columns
from piper.verbs import columns
from piper.verbs import count
from piper.verbs import counts
from piper.verbs import head
from piper.verbs import select
from piper.verbs import where
from piper.verbs import summarise
from piper.verbs import info
from piper.verbs import relocate
from piper.verbs import pivot_table
from piper.verbs import tail
from piper.verbs import trim
from piper.verbs import has_special_chars
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


# get_sample_s1 {{{1
@pytest.fixture
def get_sample_s1():

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
def test_head_with_series(get_sample_s1):
    """
    """
    s1 = get_sample_s1

    expected = (4,)
    actual = head(s1).shape
    assert expected == actual

# test_head_with_dataframe {{{1
def test_head_with_dataframe(get_sample_df2):
    """
    """
    df = get_sample_df2

    expected = (4, 2)
    actual = head(df).shape
    assert expected == actual

# test_head_with_columns_function {{{1
def test_head_with_columns_function(get_sample_df1):
    """
    """
    df = get_sample_df1

    expected = (4, 4)

    actual = head(df[columns(df, regex='order|dates|values',
        astype='list')]).shape

    assert expected == actual

# test_tail_with_series {{{1
def test_tail_with_series(get_sample_s1):
    """
    """
    s1 = get_sample_s1

    expected = (4,)
    actual = tail(s1).shape
    assert expected == actual

# test_tail_with_dataframe {{{1
def test_tail_with_dataframe(get_sample_df2):
    """
    """
    df = get_sample_df2

    expected = (4, 2)
    actual = tail(df).shape
    assert expected == actual



# test_relocate_single_column {{{1
def test_relocate_single_column(get_sample_df2):
    """
    """
    df = get_sample_df2
    df = relocate(df, 'regions', loc='first')

    expected = ['regions', 'ids']
    actual = df.columns.values.tolist()
    assert expected == actual

# test_relocate_single_column_last_column {{{1
def test_relocate_single_column_last_column(get_sample_df2):
    """
    """
    df = get_sample_df2
    df = relocate(df, 'regions', loc='last')

    expected = ['ids', 'regions']
    actual = df.columns.values.tolist()
    assert expected == actual


# test_relocate_multi_column_first {{{1
def test_relocate_multi_column_first(get_sample_df1):
    """
    """
    df = get_sample_df1
    df = relocate(df, ['dates', 'regions', 'countries'], loc='first')

    expected = ['dates', 'regions', 'countries', 'order_dates',
                'ids', 'values_1', 'values_2']

    actual = df.columns.tolist()
    assert expected == actual

# test_relocate_multi_column_last {{{1
def test_relocate_multi_column_last(get_sample_df1):
    """
    """
    df = get_sample_df1
    df = relocate(df, ['dates', 'regions', 'countries'], loc='last')

    expected = ['order_dates', 'ids', 'values_1', 'values_2',
                'dates', 'regions', 'countries' ]
    actual = df.columns.values.tolist()
    assert expected == actual

# test_relocate_multi_column_before {{{1
def test_relocate_multi_column_before(get_sample_df1):
    """
    """
    df = get_sample_df1
    df = relocate(df, ['dates', 'regions', 'countries'], loc='before',
                ref_column='values_1')

    expected = ['order_dates', 'ids', 'dates', 'regions',
                'countries', 'values_1', 'values_2']
    actual = df.columns.values.tolist()
    assert expected == actual

# test_relocate_multi_column_after {{{1
def test_relocate_multi_column_after(get_sample_df1):
    """
    """
    df = get_sample_df1
    df = relocate(df, ['dates', 'regions', 'countries'], loc='after',
                ref_column='order_dates')

    expected = ['order_dates', 'dates', 'regions', 'countries',
                'ids', 'values_1', 'values_2']
    actual = df.columns.values.tolist()
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
def test_trim_blanks(get_sample_df7):
    """
    """
    df = get_sample_df7

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
def test_trim_blanks_duplicate_column_name(get_sample_df1):
    """
    """
    df = get_sample_df1
    df.columns = ['dates', 'order_dates', 'regions', 'regions', 'ids', 'values_1', 'values_2']

    df2 = trim(df)

    expected = ['dates', 'order_dates', 'regions', 'regions2', 'ids', 'values_1', 'values_2']
    actual = df2.columns.tolist()
    assert expected == actual

# test_columns_as_list {{{1
def test_columns_as_list(get_sample_df2):

    df = get_sample_df2

    expected = ['ids', 'regions']
    actual = columns(df, astype='list')

    assert expected == actual

# test_columns_as_dict {{{1
def test_columns_as_dict(get_sample_df2):

    df = get_sample_df2

    expected = {'ids': 'ids', 'regions': 'regions'}
    actual = columns(df, astype='dict')

    assert expected == actual

# test_columns_as_dataframe {{{1
def test_columns_as_dataframe(get_sample_df2):

    df = get_sample_df2

    expected = pd.DataFrame(['ids', 'regions'], columns=['column_names'])
    actual = columns(df, astype='dataframe')

    assert_frame_equal(expected, actual)

# test_columns_as_series {{{1
def test_columns_as_series(get_sample_df2):

    df = get_sample_df2

    expected = pd.Series(['ids', 'regions'], index=range(2), name='column_names')
    actual = columns(df, astype='series')

    assert_series_equal(expected, actual)

# test_columns_as_text {{{1
def test_columns_as_text(get_sample_df2):

    df = get_sample_df2

    expected = "['ids', 'regions']"
    actual = columns(df, astype='text')

    assert expected == actual

# test_columns_as_regex {{{1
def test_columns_as_regex(get_sample_df1):

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

# test_select_no_parms {{{1
def test_select_no_parms(get_sample_df1):

    df = get_sample_df1
    df = select(df)

    expected = ['dates', 'order_dates', 'countries',
                'regions', 'ids', 'values_1', 'values_2']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_str_contains {{{1

    df = get_sample_df1
    df = select(df, df.columns.str.startswith('values'))

    expected = ['values_1', 'values_2']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_slice {{{1
def test_select_slice(get_sample_df1):

    df = get_sample_df1
    df = select(df, slice('countries', 'values_1'))

    expected = ['countries', 'regions', 'ids', 'values_1']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_column_list {{{1
def test_select_column_list(get_sample_df1):

    df = get_sample_df1
    df = select(df, ['order_dates'])

    expected = ['order_dates']
    actual = df.columns.tolist()

    assert expected == actual


# test_select_columns_list {{{1
def test_select_columns_list(get_sample_df1):

    df = get_sample_df1
    df = select(df, ['order_dates', 'countries'])

    expected = ['order_dates', 'countries']
    actual = df.columns.tolist()

    assert expected == actual

# test_select_column_str {{{1
def test_select_column_str(get_sample_df1):

    df = get_sample_df1
    df = select(df, 'order_dates')

    expected = ['order_dates']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_excluding_column_str {{{1
def test_select_excluding_column_str(get_sample_df1):

    df = get_sample_df1
    df = select(df, '-order_dates')

    expected = ['dates', 'countries', 'regions',
                'ids', 'values_1', 'values_2']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_excluding_column_list {{{1
def test_select_excluding_column_list(get_sample_df1):

    df = get_sample_df1
    df = select(df, ['-order_dates'])

    expected = ['dates', 'countries', 'regions',
                'ids', 'values_1', 'values_2']

    actual = df.columns.tolist()

    assert expected == actual


# test_select_columns {{{1
def test_select_columns(get_sample_df1):

    df = get_sample_df1
    df = select(df, ['dates', 'order_dates'])

    expected = ['dates', 'order_dates']
    actual = df.columns.tolist()

    assert expected == actual


# test_info {{{1
def test_info(get_sample_df5):

    df = get_sample_df5

    expected = (2, 6)
    actual = info(df).shape

    assert expected == actual

# test_info_with_dupes {{{1
def test_info_with_dupes(get_sample_df5):

    df = get_sample_df5

    expected = (2, 7)
    actual = info(df, include_dupes=True).shape

    assert expected == actual

# test_info_with_na_cols {{{1
def test_info_with_na_cols(get_sample_df5):

    df = get_sample_df5

    expected = (0, 6)
    actual = info(df, filter_na_cols=True).shape

    assert expected == actual

# test_count {{{1
def test_count(get_sample_df5):

    df = get_sample_df5

    expected = (1, 2)
    actual = count(df, 'bgy56icnt').shape

    assert expected == actual

# test_count_series {{{1
def test_count_series(get_sample_df5):

    df = get_sample_df5

    expected = (1, 2)
    actual = count(df.bgy56icnt).shape

    assert expected == actual

# test_count_sort {{{1
def test_count_sort(get_sample_df5):

    df = get_sample_df5

    expected = (1, 2)
    actual = count(df, 'bgy56icnt', sort_values=True).shape

    assert expected == actual

# test_count_with_total {{{1
def test_count_with_total(get_sample_df5):

    df = get_sample_df5

    expected = (2, 2)
    actual = count(df, 'bgy56icnt', add_total=True, sort_values=True).shape

    assert expected == actual

# test_count_with_total_percent_cum_percent {{{1
def test_count_with_total_percent_cum_percent(get_sample_df5):

    df = get_sample_df5

    expected = (2, 4)
    actual = count(df, 'bgy56icnt', add_total=True, sort_values=True,
                   percent=True, cum_percent=True).shape

    assert expected == actual


# test_counts_single_column {{{1
def test_counts_single_column(get_sample_df1):

    df = get_sample_df1

    expected = (4, 1)
    actual = counts(df, 'regions').shape

    assert expected == actual


# test_counts_multi_column {{{1
def test_counts_multi_column(get_sample_df1):

    df = get_sample_df1

    query = "regions == 'East' and countries.isin(['Italy'])"
    df = df.query(query)

    expected = (1, 3)
    actual = counts(df, ['regions', 'countries']).reset_index().shape

    assert expected == actual


# test_counts_multi_column_with_percent {{{1
def test_counts_multi_column_with_percent(get_sample_df1):

    df = get_sample_df1
    df = df.query("regions == 'East' and countries.isin(['Italy'])")
    df = counts(df, ['regions', 'countries'], percent=True, cum_percent=True)

    expected = 100.0
    actual = df.loc[('East', 'Italy'), 'cum %']

    assert expected == actual


# test_counts_multi_column_with_cum_percent_threshold {{{1
def test_counts_multi_column_with_cum_percent_threshold(get_sample_df1):

    df = get_sample_df1
    df = df.query("regions == 'East'")
    df = counts(df, ['regions', 'countries'],
                percent=True,
                cum_percent=True,
                threshold=81)

    expected = (5, 3)
    actual = df.shape

    assert expected == actual


# test_where {{{1
def test_where(get_sample_df1):

    df = get_sample_df1

    expected = (1, 7)
    actual = where(df, """regions == 'East' and countries == 'Spain' and values_1 == 29""")

    assert expected == actual.shape

# test_summarise_default{{{1
def test_summarise_default(get_sample_df1):

    df = get_sample_df1

    expected = (7, )
    actual = summarise(df)

    assert expected == actual.shape


# test_summarise_value {{{1
def test_summarise_value(get_sample_df1):

    df = get_sample_df1

    expected = (1, )
    actual = summarise(df, {'values_1': 'sum'})

    assert expected == actual.shape


# test_add_formula {{{1
def test_add_formula(get_sample_df2):

    df = get_sample_df2

    formula = '=CONCATENATE(A{row}, B{row}, C{row})'
    add_formula(df, column_name='X7', loc=2, formula=formula, inplace=True)

    expected = (5, )

    assert expected == df.X7.shape

# test_add_formula_pos_last {{{1
def test_add_formula_pos_last(get_sample_df2):

    df = get_sample_df2

    formula = '=CONCATENATE(A{row}, B{row}, C{row})'
    add_formula(df, column_name='X7', loc='last',
                formula=formula, inplace=True)

    assert df.columns[-1] == 'X7'


# test_add_formula_inplace_false {{{1
def test_add_formula_inplace_false(get_sample_df2):

    df = get_sample_df2

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


# test_join_df {{{1
def test_join_df(get_sample_df6):
    """
    """
    df, df2 = get_sample_df6

    df_merge = left_join(df, df2, left_on=['column_A'],
                                right_on=['column_A'])

    expected = (6, 3)
    actual = df_merge.shape

    assert expected == actual

# test_resample_groupby {{{1
def test_resample_groupby(get_sample_df1):
    """
    """

    df = get_sample_df1

    rule = 'Q'
    grouper = pd.Grouper(key='dates', freq=rule)

    index = [grouper]
    g1 = resample_groupby(df, index=index, grouper=grouper, rule=rule)

    # Tidy axis labels
    g1.index.name = 'Date period'
    g1.columns = ['Totals1', 'Totals2']

    assert g1.loc['2020 Mar', 'Totals1'] > 0

# test_resample_multi_grouper_groupby {{{1
def test_resample_multi_grouper_groupby():
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

    rule = 'D'
    grouper = pd.Grouper(key='effective', freq=rule)
    grouper2 = pd.Grouper(key='expired', freq=rule)
    multi_index = ['country', grouper, grouper2, 'prices']

    g1 = resample_groupby(df, multi_index, [grouper, grouper2], rule)

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

    rule = 'D'
    grouper = pd.Grouper(key='effective', freq=rule)
    multi_index = ['country', grouper, 'prices']

    g1 = resample_groupby(df, multi_index, grouper, rule)

    expected = (4, 1)
    actual = g1.shape

    assert expected == actual

# test_pivot_table {{{1
def test_pivot_table(get_sample_df1):
    """
    """

    df = get_sample_df1

    pv = pivot_table(df, index=['countries', 'regions'], values='values_1')
    pv.rename(columns={'values_1': 'totals'}, inplace=True)

    expected = 6507.9683290565645
    actual = pv.totals.sum()

    assert expected == actual

# test_pivot_table_sort_ascending_false {{{1
def test_pivot_table_sort_ascending_false(get_sample_df1):
    """
    """

    df = get_sample_df1

    pv = pivot_table(df, index=['countries'], values='values_1')
    pv.sort_values(by='values_1', ascending=False, inplace=True)

    expected = (8, 1)
    actual = pv.shape

    assert expected == actual

# test_pivot_name_error {{{1
def test_pivot_name_error(get_sample_df1):
    """
    Should send log message regarding invalid key/name
    pv object should be None and generate KeyError
    """

    df = get_sample_df1

    with pytest.raises(KeyError):
        pv = pivot_table(df, index=['countries_wrong_name'], values='values_1')
        pv.sort_values(by='values_1', ascending=False, inplace=True)
        pv.shape

# test_pivot_percent_calc {{{1
def test_pivot_percent_calc(get_sample_df1):
    """
    """

    df = get_sample_df1

    pv = pivot_table(df, index=['countries', 'regions'], values='values_1')
    pv.rename(columns={'values_1': 'totals'}, inplace=True)
    pv.sort_values(by='totals', ascending=False, inplace=True)
    pv['%'] = pv.totals.apply(lambda x: x*100/pv.totals.sum())

    expected = 2.874168873331256
    actual = pv.loc[('Norway','East'), '%']

    assert expected == actual

# test_pivot_cum_percent_calc {{{1
def test_pivot_cum_percent_calc(get_sample_df1):
    """
    """

    df = get_sample_df1

    pv = pivot_table(df, index=['countries', 'regions'], values='values_1')
    pv.rename(columns={'values_1': 'totals'}, inplace=True)
    pv.sort_values(by='totals', ascending=False, inplace=True)
    pv['%'] = pv.totals.apply(lambda x: x*100/pv.totals.sum())
    pv['cum %'] = pv['%'].cumsum()

    expected = 79.67310369428336
    actual = pv.loc[('Sweden', 'West'), 'cum %']

    assert expected == actual

# test_resample_pivot_multi_grouper {{{1
def test_resample_pivot_multi_grouper(get_sample_df1):
    """
    """
    df = get_sample_df1

    rule = 'Q'
    grouper = pd.Grouper(key='dates', freq=rule)
    grouper2 = pd.Grouper(key='order_dates', freq=rule)

    index = [grouper, grouper2, 'regions', 'ids']
    p2 = resample_pivot(df, index=index, grouper=[grouper, grouper2],
                        rule=rule)

    assert p2.loc[('2020 Mar', '2020 Mar', 'East', 'A'), 'values_1'] > 0

# test_resample_pivot_single_grouper {{{1
def test_resample_pivot_single_grouper(get_sample_df1):
    """
    """
    df = get_sample_df1

    rule = 'Q'
    grouper = pd.Grouper(key='dates', freq=rule)

    index = [grouper, 'regions', 'ids']
    p2 = resample_pivot(df, index=index, grouper=grouper, rule=rule)

    assert p2.loc[('2020 Mar', 'East', 'A'), 'values_1'] > 0

# test_calc_grouped_value {{{1
def test_calc_grouped_value(get_sample_df1):
    """
    """

    df = get_sample_df1

    # % and total group contribution (values_1)
    index = ['countries', 'regions']

    x = lambda x: x.sum()

    subset_df = df[['countries', 'regions', 'ids', 'values_1', 'values_2']].copy()

    gx = add_group_calc(subset_df, column='group % contribution',
                       value='values_2', index=index)

    gx = add_group_calc(subset_df, column='Total Grouped Value',
                        value='values_2', index=index, function=x)

    gx.set_index(['countries', 'regions', 'ids'], inplace=True)

    expected = (367,4)
    actual = gx.shape

    assert expected == actual

# test_has_special_chars {{{1
def test_has_special_chars(get_sample_df1):
    """
    """
    def generate(random_chars=10):
        alphabet="0123456789abcdefghijklmnopqrstuvwxyz!$%()"

        r = random.SystemRandom()
        chars = ''.join([r.choice(alphabet) for i in range(random_chars)])

        return chars

    dx = pd.DataFrame([generate() for x in range(30)], columns=['col'])

    df_count = count(has_special_chars(dx, 'col'), 'item_special_chars')

    assert df_count.shape == (2, 2)

# test_duplicated {{{1
def test_duplicated(get_sample_df5):
    """
    """
    df = get_sample_df5

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
def test_duplicated_first_position(get_sample_df5):
    """
    """
    df = get_sample_df5

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
def test_duplicated_last_position(get_sample_df5):
    """
    """
    df = get_sample_df5

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
def test_duplicated_inplace_true(get_sample_df5):
    """
    """
    df = get_sample_df5

    duplicated_fields = ['bgy56icnt', 'bgz56ccode']

    duplicated(df[duplicated_fields],
            subset=duplicated_fields,
            keep='first', sort=True,
            column='dupe',
            loc='first')

    assert_frame_equal(df, df.copy(deep=True))

# test_duplicated_data_raise_col_position_error {{{1
def test_duplicated_data_raise_col_position_error(get_sample_df5):
    """
    """
    df = get_sample_df5
    duplicated_fields = ['bgy56icnt', 'bgz56ccode']

    with pytest.raises(Exception):
        assert duplicated(df[duplicated_fields],
                          subset=duplicated_fields, keep='first',
                          sort=True, insert_column=True,
                          column_name='dupe', column_pos=6) is None

# test_to_tsv_with_data {{{1
def test_to_tsv_with_data(get_sample_orders_01):
    """
    Write sample dataframe, no value is return from to_tsv
    """
    file_name = 'piper/temp/to_tsv_with_data.tsv'
    df = pd.DataFrame(get_sample_orders_01)

    expected = None
    actual = to_tsv(df, file_name, sep='\t')

    assert expected == actual


# test_read_csv_with_data {{{1
def test_read_csv_with_data(get_sample_orders_01):
    """
    """
    file_name = 'piper/temp/to_tsv_with_data.tsv'
    df_json = pd.DataFrame(get_sample_orders_01)

    df = read_csv(file_name, sep='\t')

    expected = df_json.shape
    actual = df.shape

    assert expected == actual
