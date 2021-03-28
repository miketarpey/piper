import pypyodbc as pyodbc
import cx_Oracle
import pandas as pd
from psycopg2 import connect as pg_connect
from piper.utils import get_config
from piper.io import read_sql

import logging


logger = logging.getLogger(__name__)


def connections(file_name=None, return_type='dataframe'):
    ''' get all available defined database environments

    Parameters
    ----------
    file_name
        json formatted connections file (similar to tnsnames.ora)
        if None (default) uses ../src/config.json
    return type
        return object type: 'dataframe'(default) 'dictionary'

    Returns
    -------
    dictionary or dataframe

    Examples
    --------

    .. code-block::

        df = connections(return_type='dataframe')
        dict_config = connections(return_type='dictionary')

    '''
    if file_name == None:
        default_config = get_config('config.json')
        file_name = default_config['connections']['location']

    config = get_config(file_name, info=False)
    logger.debug(config)

    if return_type == 'dictionary':
        return config

    if return_type == 'dataframe':
        df = (pd.DataFrame(config).T).fillna('')

        if 'pw' in df.columns:
           df = df.drop(columns=['pw'])

        lowercase_cols = ['schema', 'sid', 'user']
        for col in lowercase_cols:
            if col in df.columns:
                df[col] = df[col].str.lower()

    return df


def connect(connection=None, connection_type=None, file_name=None):
    ''' Return database connection

    Examples
    --------

    .. code-block::

        con = connect('beles8', connection_type='ibm')
        con = connect('orders_db', connection_type='postgres')
        con, schema, schema_ctl = connect('JDE8EPA')

    For list of available connections type:

    .. code-block::

        connections()

    Parameters
    ----------
    connection
        connection name e.g. 'beles8', 'jde_prod'
        stored in connections.json in default ap folder
    connection_type
        values - 'postgres', 'ibm', 'oracle'

    Returns
    -------
    If connection_type is = '10.21.1.15' then
        return iSeries connection
    else
        oracle connection (con, schema, schema_ctl)
    '''
    if connection_type is None:
        connection_type = 'postgres'

    config = connections(file_name, return_type='dictionary')

    try:
        connection_parms = config[connection]
    except KeyError as _:
        connection_parms = None
        raise KeyError(f'invalid connection {connection}')

    # No tests since I no longer have access to B2 or B8 ;).
    if connection_type == 'ibm':
        con = _get_iseries_con(connection_parms)
        return con
    elif connection_type == 'postgres':
        con = _get_postgres_con(connection_parms)
        return con
    else:
        con, schema, schema_ctl = _get_oracle_con(connection_parms)
        return con, schema, schema_ctl


def _get_iseries_con(connection_parms):
    ''' Return ibm connection for given system.

    Examples
    --------

    .. code-block::

        con = _get_iseries_con('beles8')

    Parameters
    ----------
    connection_parms
        connection parameters

    Returns
    -------
    ibm odbc connection
    '''

    driver = 'IBM i Access ODBC Driver'
    # No tests since I no longer have access to B2 or B8 ;).
    host = connection_parms.get('host')
    user = connection_parms.get('user')
    pw   = connection_parms.get('pw')
    con = pyodbc.connect(driver=driver, system=host, uid=user, pwd=pw)

    logger.info(f"Connection: {host}, user:{user}")
    logger.info(f'Connected: {con.connected == 1}')

    return con


def _get_postgres_con(connection_parms):
    ''' Return postgres connection for given system.

    Examples
    --------

    .. code-block::

        con = _get_postgres_con('beles8')

    Parameters
    ----------
    connection_parms
        connection parameters

    Returns
    -------
    postgres odbc connection
    '''
    host = connection_parms.get('host')
    db   = connection_parms.get('schema')
    user = connection_parms.get('user')
    pw   = connection_parms.get('pw')
    con = pg_connect(host=host, dbname=db, user=user, password=pw)

    logger.info(f"Connection: {host}, user:{user}")

    return con


def _get_oracle_con(connection_parms='JDE8EPA'):
    ''' Return Oracle connection and schema, schema_ctl.

    Examples
    --------

    .. code-block::

        con, schema, schema_ctl = _get_oracle_con('JDE8EPA')

    Parameters
    ----------
    connection_parms
        connection_parms environment key connection (values)
        stored in connections.json in default ap folder

    Returns
    -------
    Oracle connection, schema and schema (control) name strings
    '''
    host    = connection_parms.get('host')
    port    = connection_parms.get('port')
    sid     = connection_parms.get('sid')
    service = connection_parms.get('service')
    user    = connection_parms.get('user')
    pw      = connection_parms.get('pw')

    logger.debug(f"Connection: {host}, {port}, {sid}, {svc}")
    logger.info(f"Connection: {host}, {port}")

    dsn_tns = cx_Oracle.makedsn(host=host, port=port, service_name=svc)
    logger.debug(dsn_tns)

    connection = cx_Oracle.connect(user, pw, dsn_tns, encoding="UTF-8")
    logger.debug(f'TNS: {connection.dsn}')
    logger.debug(f'Version: {connection.version}')

    schema = connection_parms.get('schema').lower()
    schema_ctl = connection_parms.get('schema').lower().replace('dta', 'ctl')
    logger.debug(f'Schema: {schema}')

    return connection, schema, schema_ctl
