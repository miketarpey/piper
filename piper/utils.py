import json
import logging
import pandas as pd
from pathlib import Path
from os.path import dirname, abspath

from IPython import get_ipython
from IPython.core.magic import magics_class
from .magics import PipeMagic


logger = logging.getLogger(__name__)

# get_config() {{{1
def get_config(file_name=None, info=False):
    ''' Retrieve configuration file ('config.json')

    For given filename, retrieve configuration details as a dictionary.
    Odbc user/passwords connections are stored in file 'connections.json'

    Search strategy is as follows:
        1. Look in current notebook directory
        2. Look in home/piper
        3. Look in package directory (failsafe)

    Parameters
    ----------
    filename
        if None then return {}
    info
        Default True. Return and display pd.DataFrame (via notebook)

    Returns
    -------
    config as a dictionary
    '''
    config = None
    current_dir = Path.cwd()
    piper_dir = Path.home() / 'piper'
    package_dir = Path(abspath(dirname(__file__))) / 'config'

    search_list = [current_dir, piper_dir, package_dir]

    for location in search_list:

        qual_file = location / file_name
        try:
            with open(qual_file.as_posix(), 'r') as f:
                config = json.load(f)

            if info:
                logger.info(f'config file: {qual_file}')

            break

        except (FileNotFoundError, TypeError) as _:
            if info:
                logger.info(f'config file: {qual_file} not found')
            continue

    return config


# get_dict_df() {{{1
def get_dict_df(dict, colnames=None):
    ''' get dataframe from a dictionary

    Parameters
    ----------
    dict
        a dictionary containing values to be converted
    colnames
        list of columns to subset the data

    Returns
    -------
    pandas df oriented by index and transposed
   '''

    if colnames is None:
        df = pd.DataFrame.from_dict(dict, orient='index')
        df.fillna('', inplace=True)
    else:
        df = pd.DataFrame.from_dict(dict, orient='index', columns=colnames)
        logger.debug(df)

    return df


# is_type() {{{1
def is_type(value, type_=str):
    ''' is type is useful when determining if a dataframe column
    contains a certain datatype (e.g. str, int, float)

    Parameters
    ----------
    value
        value to be checked
    type
        data type (e.g. str, int, float etc.)

    Examples
    --------
    df[df.bgexdj.apply(is_type)]

    Returns
    -------
    boolean
    '''
    if isinstance(value, type_):
        return True

    return False


# register_magic() {{{1
def register_magic():
    '''Register pipe magic '''

    ip = get_ipython()
    ip.register_magics(PipeMagic)

    return ip
