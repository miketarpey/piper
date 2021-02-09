import json
import logging
import pandas as pd
from shutil import copy
from pathlib import Path
from os.path import dirname, abspath


logger = logging.getLogger(__name__)


def get_config(file_name=None, info=False):
    ''' Given piper configuration file 'config.json' or
    odbc user/passwords connections file 'connections.json'
    retrieve dictionary contents and return

    Search strategy is as follows:
    # 1. Look in current notebook directory
    # 2. Look in home/piper
    # 3. Look in package directory (failsafe)

    Parameters
    ----------
    filename : if None then return {}

    info : True, return and display pd.DataFrame (via notebook)

    Returns
    -------
    config as a dictionary
    '''
    config = None
    current_dir = Path.cwd()
    piper_dir = Path.home() / 'piper'
    package_dir = Path(abspath(dirname(__file__))) / 'config'
    # package_dir = Path(abspath(dirname(__file__)))

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


def get_dict_df(dict, colnames=None):
    ''' get dataframe from a dictionary

    Parameters
    ----------
    dict : a dictionary containing values to be converted

    colnames : list of columns to subset the data

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
