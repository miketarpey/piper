from piper.pandas import generate_summary_df
from piper.pandas import is_type
from piper.pandas import read_sql
from piper.verbs import count
from piper.verbs import trim
from piper.test.factory import get_sample_orders_01
from piper.test.factory import get_sample_df4
from pandas._testing import assert_frame_equal
from pandas._testing import assert_series_equal
import random
import json
import numpy as np
import pandas as pd
import pytest


# test_read_sql_valid_info_true {{{1
@pytest.mark.skip(reason="no way of currently testing this")
def test_read_sql_valid_info_true():
    """
    """
    env = 'JDE9E2P'
    con, schema, schema_ctl = connect(env)

    sql = ''' select * from eudta.f0006 where rownum=1'''
    df = read_sql(sql, sql_info=False, con=con, info=True)

    expected = (1, 152)
    actual = df.shape
    assert expected == actual


# test_read_sql_valid_info_false
@pytest.mark.skip(reason="no way of currently testing this")
def test_read_sql_valid_info_false():
    """
    """
    env = 'JDE9E2P'
    con, schema, schema_ctl = connect(env)

    sql = ''' select * from eudta.f0006 where rownum=1'''
    df = read_sql(sql, sql_info=True, con=con, info=False)

    expected = (1, 152)
    actual = df.shape
    assert expected == actual


# test_read_sql_valid_sql_info_true
@pytest.mark.skip(reason="no way of currently testing this")
def test_read_sql_valid_sql_info_true():
    """
    """
    env = 'JDE9E2P'
    con, schema, schema_ctl = connect(env)

    sql = ''' select * from eudta.f0006 where rownum=1'''
    df = read_sql(sql, sql_info=True, con=con, info=True)

    expected = (1, 152)
    actual = df.shape
    assert expected == actual


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

# test_generate_summary_df {{{1
def test_generate_summary_df(get_sample_df4):
    """
    """
    df, df2 = get_sample_df4

    datasets = [('1st dataset', df), ('2nd dataset', df2)]
    summary_df = generate_summary_df(datasets, title='Summary',
                                     col_total='Total records',
                                     add_grand_total=True,
                                     grand_total='Grand total')
    expected = 24
    actual = summary_df.loc['Grand total', 'Total records']

    assert expected == actual