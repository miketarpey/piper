from piper.configure import get_config
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# create_nb_proj {{{1
def create_nb_proj(project='project', description='notebook project',
                   relative_to='.', ):
    '''
    Create notebook project.


    Parameters:
    -----------
    project: str project folder name.

    description: str project description.

    relative_to: str relative path to current folder (default '.')


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

    logger.info(f'{project} folder completed.')


# create_nb_folders {{{1
def create_nb_folders(project='project', relative_to='.', ):
    '''
    Create notebook folders based on config.json in
    project folder specified.


    Parameters:
    -----------
    project: str project folder name.

    relative_to: str relative path to current folder
                 (default '.')


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
            logger.info(f'Created subfolder: {qual_folder}')
        except FileExistsError as e:
            logger.info(e)
