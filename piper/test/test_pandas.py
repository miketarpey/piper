from piper.verbs import summary_df
from piper.utils import is_type
from piper.pandas import read_sql
from piper.factory import bad_quality_orders
from piper.factory import get_sample_df4
from pandas._testing import assert_frame_equal
from pandas._testing import assert_series_equal
import pytest


@pytest.fixture
def sample_orders_01():
    return bad_quality_orders()


@pytest.fixture
def sample_df4():
    return get_sample_df4()


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
def test_generate_summary_df(sample_df4):
    """
    """
    df, df2 = sample_df4

    datasets = [('1st dataset', df), ('2nd dataset', df2)]
    summary = summary_df(datasets, title='Summary',
                            col_total='Total records',
                            add_grand_total=True,
                            grand_total='Grand total')
    expected = 24
    actual = summary.loc['Grand total', 'Total records']

    assert expected == actual
