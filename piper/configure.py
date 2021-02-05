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


def load_nb_config(config_file: str=None):
    ''' Load parameter/config file

    Examples
    --------
    defaults.py
    1. Load default variables into current notebook. If no project configuration
       file, then try to load local project default file called 'config.json'.

    defaults.py "project.json"
    2. If a project configuration file passed, load that.

    '''
    if config_file is None:
        config_file = 'config.json'

    try:
        logger.info(f'Local config >> {config_file}')
        config = get_config(config_file)

    except FileNotFoundError as _:
        logger.info(_)
        return None

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


def create_nb_project(project='project', description='project description'):
    '''
    Create notebook project.

    Parameters:
    -----------

    project: str project folder name. This folder will be built
                under the main project folder (usually 'notebooks')

    description: str project description. Used to setup default project name
        when building reports (accessible in notebooks as 'project' variable)

    Returns:
    --------
    None

    '''

    # Assumptions: source directory and notebooks setup as below
    setup = {'root': Path.home() / 'Documents/notebooks',
             'src': Path.home() / 'Documents/notebooks/src'}

    # Retrieve root directory...
    project_root = setup.get('root') / project
    logger.info(project_root)

    # Get default sub-directory configuration, from config.json
    # to build the folders.
    config_file = 'config.json'
    default_config = setup.get('src') / config_file
    project_config = get_config(default_config)
    logger.debug(project_config)

    # Build main project directory
    try:
        project_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as e:
        logger.info(e)

    for folder in project_config['folders']:
        try:
            qual_folder = project_root / folder
            qual_folder.mkdir(parents=True, exist_ok=False)
            logger.info(f'Created subfolder: {folder}')
        except FileExistsError as e:
            logger.info(e)

    # Build a project specific config.json
    project_config_json = project_root / config_file
    logger.debug(project_config_json)

    project_config['project'] = description
    project_config['meta']['project'] = description
    project_config['mail_config']['sender'] = 'someone@acme.com'

    # Write to the project default folder
    with open(project_config_json, "w") as outfile:
        json.dump(project_config, outfile)

    logger.info(f'{project_root} folder completed.')
