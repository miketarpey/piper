from nbconvert.preprocessors import CellExecutionError 
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
from piper.io import list_files
from piper.utils import get_config
import json
import logging
import nbformat
import pandas as pd
import re

from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Pattern,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# create_nb_proj {{{1
def create_nb_proj(project: str = 'project',
                   description: str = 'notebook project',
                   relative_to: str = '.',
                   sub_folders: bool = True):
    ''' Create notebook project.

    This function uses the default configuration file ('config.json') to
    create a default jupyter project folder.

    Examples
    --------

    .. code-block::
        create_nb_proj('eoin_dev')

    Parameters:
    -----------
    project
        project folder name.
    description
        project description.
    relative_to
        relative path to current folder (default '.')


    Returns:
    --------
    None
    '''
    project_path = Path(relative_to).absolute().parent / project

    # Check whether project folder exists
    if project_path.exists():
        raise FileExistsError(f'project: {project_path} already exists!')

    # create project directory
    try:
        project_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as e:
        logger.info(e)

    # Copy package default config.json and adjust as needed.
    config_file = 'config.json'
    config = get_config(config_file, info=False)

    config['project'] = description
    config['meta']['project'] = description
    logger.debug(config)

    # Write to the project default folder
    new_config = project_path / config_file

    with open(new_config, "w") as f:
        json.dump(config, f)

    if sub_folders:
        create_nb_folders(project=project,
                          relative_to=relative_to)

    logger.info(f'{project} folder created.')


# create_nb_folders {{{1
def create_nb_folders(project: str = 'project',
                      relative_to: str = '.'):
    ''' Create notebook sub folders

    For given 'project name', this function will use the default configuration
    file 'config.json' to create the project sub-folders specified.

    Parameters:
    -----------
    project
        project folder name.
    relative_to
        relative path to current folder (default '.')

    Returns:
    --------
    None
    '''
    project_path = Path(relative_to).absolute().parent / project

    qual_config_file = project_path / 'config.json'
    if not qual_config_file.exists():
        raise FileNotFoundError(f'{qual_config_file} does not exist.')

    with open(qual_config_file, "r") as f:
        config = json.load(f)

    for folder in config['folders']:
        try:
            qual_folder = project_path / folder
            qual_folder.mkdir(parents=True, exist_ok=False)
            logger.debug(f'Created subfolder: {qual_folder}')
        except FileExistsError as e:
            logger.info(e)

    logger.info(f"Created subfolders...{config['folders']}")


# execute_notebook {{{1
def execute_notebook(input_file: Path,
                     output_file: Path) -> Dict:
    ''' Execute notebook within a folder


    Parameters
    ----------
    input_file
        notebook file to be executed
    output_file
        notebook execution output file

    Returns
    -------
    a pandas dataframe
        contains filename and error(s) encountered.

    '''
    error = {}

    with open(input_file) as f:
        try:
            nb = nbformat.read(f, as_version=4)
            kernel = nb.dict()['metadata']['kernelspec']['name']

            logger.info(f'Running... {input_file.stem}')
            ep = ExecutePreprocessor(timeout=600, kernel_name=kernel)

            out = ep.preprocess(nb, {'metadata': {'path': input_file.parent}})

        except CellExecutionError:
            out = None
            msg = f'Error executing notebook".\n'
            msg += f'See notebook "{output_file}" for the traceback.'
            logger.info(msg)
            error = {'error': msg, 'file': input_file}
        finally:
            with open(output_file, mode='w', encoding='utf-8') as f:
                nbformat.write(nb, f)

    return error
# execute_notebooks_in_folder {{{1
def execute_notebooks_in_folder(input_folder: str,
                                output_folder: str = None) -> pd.DataFrame:
    ''' execute notebooks within a folder

    Examples
    --------

    .. code-block::

            nb_projects = [
            'Documents/notebooks/piper_eda_examples/',
        #     'Documents/github_repos/piper_demo/',
        #     'Documents/github_repos/fake_data/'
        ]

        for project in nb_projects:
            df = execute_notebooks_in_folder(project)

    Parameters
    ----------
    input_folder
        input folder containing notebooks to be run
    output_folder
        output folder containing execution output


    Returns
    -------
    a pandas dataframe
    '''
    home_directory = Path.home()

    if output_folder is None:
        output_folder =  'Documents/notebooks/temp/'

    files = list_files(source=input_folder, glob_pattern='*.ipynb*')

    errors = []
    for f in files:
        if not f.is_dir():
            input_file = input_folder / f
            output_file = output_folder / f
            errors.append(execute_notebook(input_file, output_file))

    df = pd.DataFrame(errors)

    if df.shape[1] > 0:
        df = df.dropna().sort_values('file').reset_index(drop=True)
        df.insert(0, 'filename', df.file.apply(lambda x: x.stem))

    return df

# nb_search {{{1
def nb_search(path: str, text: str = None) -> pd.DataFrame:
    ''' search notebooks for matching text criteria

    Examples
    --------

    .. code-block::

        paths = {
            '/home/mike/Documents/notebooks/piper_eda_examples/',
            '/home/mike/Documents/github_repos/piper_demo/'
        }

        dataframes = [nb_search(path=path, text='lambda') for path in paths]
        dataframes = pd.concat(dataframes)
        dataframes

    Parameters
    ----------
    path
        notebook directory to search
    text
        text

    Returns
    -------
    pd.DataFrame
        containing files which contain matching text

    '''
    if text is None:
        logger.info("Please enter text parameter to search")
        return

    files = list_files(path, glob_pattern='*.ipynb')

    references = []
    for filename in files:
        with open(filename) as f:
            lines = f.readlines()

            for line_no, line in enumerate(lines, start=1):
                match = re.search(text, line)
                if match:
                    references.append({
                        'path': filename.parent,
                        'file': filename.stem,
                        'line_nbr': line_no,
                        'text': line.strip()
                    })
    df = pd.DataFrame(None)
    if len(references) > 0:
        df = pd.DataFrame(references).sort_values(['path', 'file', 'line_nbr'])
        df = df.reset_index(drop=True)

    return df
