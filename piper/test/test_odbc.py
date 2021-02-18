import pytest
import pandas as pd
from piper.odbc import connections


def test_connections_file_not_found_returns_empty_dataframe():
    """
    """
    expected = True
    actual = connections('no_file.json')

    assert expected == isinstance(actual, pd.DataFrame)


def test_connections_default_returns_dataframe():
    """
    """
    df = connections(return_type='dataframe')
    assert isinstance(df, pd.DataFrame) is True



@pytest.mark.skip(reason="no way of currently testing this")
def test_get_oracle_connection_valid():
    """
    """
    config = connections(return_type='dictionary')
    env = config['JDE8EPA']
    ora_connection, schema, schema_ctl = _get_oracle_con(env)
    connect_list = ora_connection.tnsentry \
                                 .replace('(', '').replace(')', ';') \
                                 .replace(';;', ';').split(';')

    assert connect_list[1] == 'HOST=dbsJDE8EPA.aws.baxter.com'


@pytest.mark.skip(reason="no way of currently testing this")
def test_oracle_config_found():
    """
    """
    config = connections(return_type='dictionary')
    env = config['JDE81EQ']
    assert env.get('host') == 'dbsJDE81EQ.aws.baxter.com'
    assert env.get('user') == 'miket'
    assert env.get('sid') == 'JDE81EQ'


@pytest.mark.skip(reason="no way of currently testing this")
def test_oracle_schemas():
    """
    """
    expected_schema = 'eudta'
    expected_schema_ctl = 'euctl'

    config = connections(return_type='dictionary')
    env = config['JDE9E2P']
    con, schema, schema_ctl = _get_oracle_con(env)

    assert expected_schema == schema
    assert expected_schema_ctl == schema_ctl
