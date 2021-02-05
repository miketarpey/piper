from pathlib import Path
import re
import logging


logger = logging.getLogger(__name__)


def get_sql_text(file_or_text, variables, debug=False):
    ''' Get SQL text file and substitute template
        variables with (dictionary) variables passed.

    Parameters
    ----------
    file_or_text :
        file name containing sql or actual text to
        convert

    variables :
        dictionary of variables to be used for replacing
        substitutional variables.

    debug :
        True, False
        If True, provide logging/debug information

    Returns
    -------
    text string containing replaced substitutional variables
    '''

    # If the file does not exist, assume that SQL text
    # has been passed to the function, process as such.
    try:
        filename = Path(file_or_text)

        if filename.exists():
            with open(filename.as_posix(), 'r') as f:
                sql = f.read()

            sql = _get_sql_text(sql, variables, debug=False)

            if debug:
                logger.info(f'{file_or_text}')
                logger.info(variables)
                logger.info(sql)

            return sql

        if isinstance(file_or_text, str):
            sql = _get_sql_text(file_or_text, variables, debug=False)
            return sql

    except OSError as _:
        sql = _get_sql_text(file_or_text, variables, debug=debug)

        return sql


def _get_sql_text(sql, variables, debug=False):
    ''' Get SQL text string and substitute statement
        variables with dictionary variables passed.

    Parameters
    ----------
    sql       : sql text

    variables : dictionary of variables to be used for
                replacing substitutional variables.

    debug     : True, False
                if True, provide logging/debug information

    Returns
    -------
    text string containing replaced substitutional variables
    '''

    for key, variable in variables.items():
        check_key = '{' + f'{key}' + '}'
        if isinstance(variable, list):
            variable = ', '.join(map(lambda x: f"'{x}'", variable))
        elif isinstance(variable, int):
            variable = str(variable)

        sql = re.sub(pattern=check_key, repl=variable, string=sql, flags=re.I)

    if debug:
        logger.info(variables)
        logger.info(sql)

    return sql


def set_sql_file(from_file=None, to_file=None, template_values=True):
    ''' Set SQL file text to 'template' substitutional values

    Example
    -------

    Set/replace existing SQL text file to template format:

    project_dir = Path.home() / 'documents'

    from_file = project_dir / 'Scratchpad.sql'
    to_file = project_dir / 'Scratchpad_revised.sql'
    set_sql_file(from_file, to_file, template_values=True)


    To convert a 'templated' sql text file to use hardcoded schema values:

    from_file = project_dir / 'Scratchpad_revised.sql'
    to_file = project_dir / 'Scratchpad_revised2.sql'

    template_values = {'{schema}': 'eudta', '{schema_ctl}': 'euctl'}
    set_sql_file(from_file, to_file, template_values)


    Parameters
    ----------
    from_file : source file containing sql text to be converted

    to_file : target file to contain substituted text

    template_values :
        True (convert sql text to use 'template' variables)

        or,

        string value (a dictionary of key/values each containing
        a regex statement and replacement values

    Returns
    -------
    None
    '''
    if isinstance(to_file, Path):
        to_file.as_posix()

    with open(from_file, 'r') as f:
        sql = f.read()

    sql = set_sql_text(sql, template_values=template_values)

    with open(to_file, 'w') as f:
        f.write(sql)


def set_sql_text(sql, template_values=True):
    ''' Set SQL text values to 'template' values

    Example
    -------
    sql = set_sql_text(sql, template_values=True)


    Parameters
    ----------
    sql : sql text string

    template_values :
        True (convert sql text to use 'template' variables)

        or,

        string value (a dictionary of key/values each containing
        a regex statement and replacement values

    Returns
    -------
    sql (rendered text)

    '''

    if template_values is True:
        replacements = {
            '(proddta|eupldta|euqadta|eutestdta|eudta|eudta2)': '{schema}',
            '(prodctl|euplctl|euqactl|eutestctl!euctl|euctl2)': '{schema_ctl}'
        }
    else:
        replacements = template_values

    for key, value in replacements.items():

        if template_values is not True:
            key = '{' + f'{key}' + '}'

        sql = re.sub(pattern=key, repl=value, string=sql, flags=re.I)

    return sql
