import re
import logging
from os.path import join
from pathlib import Path


logger = logging.getLogger(__name__)


# create_table() {{{1
def create_table(df, tablename=None, data_types=None, not_null=False, info=False):
    ''' SQL - generate 'create table' sql stmt(s)

    Function to build sql database table for supplied dataframe.

    Parameters
    ----------
    df
        dataframe containing rows to be inserted
    table_name
        target database table name
    data_types:
        dictionary containing corresponding data types between pandas and target
        database.

        If None, function will apply 'default' from/to data type values.
    not null
        default False. If True, insert 'not null' in SQL column statement(s)

    Returns
    -------
    sql
        sql text containing insert statement(s)
    '''
    if tablename is None:
        tablename = 'table'

    if data_types is None:
        data_types = {'object': 'text', 'int32': 'integer',
                      'int64': 'bigint', 'float64': 'decimal',
                      'datetime64[ns]': 'date'}

    null = ' not_null' if not_null else ''

    columns = []
    for column_name, _type in df.dtypes.items():
        sql_data_type = data_types.get(_type.name, "<< unknown >>")

        columns.append(f'\n{column_name} {sql_data_type}{null}')

    sql = f"create table {tablename} ({', '.join(columns)}\n) "

    if info:
        logger.info(sql)

    return sql


# insert() {{{1
def insert(df, tablename=None, info=False):
    """ SQL - Generate mass insert sql stmts

    Function to build sql insert statements for supplied dataframe.

    Note: lambda 'func' defined to build row values delimitting
          str values using apostrophes


    Parameters
    ----------
    df
        pandas dataframe containing rows to be inserted
    table_name
        target database table name


    Returns
    -------
    sql
        sql text containing insert statement(s)
    """
    columns = ', '.join(df.columns.tolist())

    def check_valid(value):
        '''
        '''

        if isinstance(value, int):
            return f"'{value}'"
        if isinstance(value, float):
            return f"'{value}'"
        else:
            if isinstance(value, str):
                if re.search("[']", value):
                    value = re.sub("[']", "\\'", value)
                    return f"E'{value}'"

        return f"'{value}'"

    func = lambda x: f"""'{x}'"""
    rows = ('('+', '.join(map(check_valid, r)) +')' for r in df.values)
    values = ', '.join(rows)

    sql = f"insert into {tablename} ({columns}) values {values}"

    if info:
        logger.info(sql)

    return sql


# get_sql_text() {{{1
def get_sql_text(file_or_text, variables, debug=False):
    ''' Get SQL text file and substitute template
        variables with (dictionary) variables passed.

    Parameters
    ----------
    file_or_text
        file name containing sql or actual text to convert
    variables
        dictionary of variables to be used for replacing
        substitutional variables.
    debug
        If True, provide logging/debug information

    Returns
    -------
    text string
        containing replaced substitutional variables
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


# set_sql_file {{{1
def set_sql_file(from_file=None, to_file=None, template_values=True):
    ''' Set SQL file text to 'template' substitutional values

    Examples
    --------

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
    from_file
        source file containing sql text to be converted
    to_file
        target file to contain substituted text
    template_values
        - True (convert sql text to use 'template' variables) or,
        - string value (a dictionary of key/values each containing a regex
          statement and replacement values

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


# set_sql_text {{{1
def set_sql_text(sql, template_values=True):
    ''' Set SQL text values to 'template' values

    Examples
    --------
    sql = set_sql_text(sql, template_values=True)


    Parameters
    ----------
    sql
        sql text string
    template_values
        - True (convert sql text to use 'template' variables) or,
        - string value (a dictionary of key/values each containing a regex
          statement and replacement values

    Returns
    -------
    sql
        rendered text
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


# _get_sql_text {{{1
def _get_sql_text(sql, variables, debug=False):
    ''' Get SQL text string and substitute statement
        variables with dictionary variables passed.

    Parameters
    ----------
    sql
        sql text

    variables
        dictionary of variables to be used for replacing substitutional
        variables.
    debug
        If True, provide logging/debug information.

    Returns
    -------
    text
        string containing replaced substitutional variables
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
