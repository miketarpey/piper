from piper.verbs import summary_df
from piper.utils import is_type
from piper.pandas import read_sql
from piper.factory import bad_quality_orders
import pandas as pd
from pandas._testing import assert_frame_equal
from pandas._testing import assert_series_equal
import pytest


@pytest.fixture
def sample_orders_01():
    return bad_quality_orders()


@pytest.fixture
def sql():
    return ''' select * from eudta.f0006 where rownum=1'''


# test_read_sql_valid_info_true {{{1
@pytest.mark.skip(reason="no way of currently testing this")
def test_read_sql_valid_info_true(sql):
    """
    """
    env = 'JDE9E2P'
    con, schema, schema_ctl = connect(env)
    df = read_sql(sql, sql_info=True, con=con, info=False)
    assert (1, 152) == df.shape


# test_read_sql_valid_info_false
@pytest.mark.skip(reason="no way of currently testing this")
def test_read_sql_valid_info_false(sql):
    """
    """
    env = 'JDE9E2P'
    con, schema, schema_ctl = connect(env)
    df = read_sql(sql, sql_info=False, con=con, info=True)
    assert (1, 152) == df.shape


# test_istype_int {{{1
def test_istype_int():

    expected = True
    actual = is_type(20, int)

    assert expected == actual

# test_istype_float {{{1
def test_istype_float():

    expected = True
    actual = is_type(45623.32, float)

    assert expected == actual

# test_istype_not_str {{{1
def test_istype_not_str():

    expected = False
    actual = is_type(30, str)

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
