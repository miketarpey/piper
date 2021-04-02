import re
import logging
from os.path import normpath, join
from pathlib import Path
from datetime import datetime
import pandas as pd

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


# get_sql_text() {{{1
def get_sql_text(file_or_text, variables, debug=False):
    ''' Get SQL text file and substitute template
        variables with (dictionary) variables passed.

    Parameters
    ----------
    file_or_text
        file name containing sql or actual text to
        convert
    variables
        dictionary of variables to be used for replacing
        substitutional variables.
    debug
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


# pipe_parser() {{{1
def pipe_parser(text: str,
                pipe_symbol: Optional[str] = None,
                info: bool = False,
                debug: bool = False) -> Tuple[str, Callable]:
    ''' piper parser - split by pipe_symbol chained expression statements

    Return completed pipeline of functions/commands, plus a boolean to determine
    whether the statement should be executed or not by the piper magic method
    (run_cmd).

    Examples
    --------

    .. code-block::

        %%piper --info
        get_sample_sales() >>
        select('-target_profit') >>
        reset_index(drop=True) >>
        head()

    is rendered by the parser as:

    .. code-block::

        (get_sample_sales()
        .pipe(select, '-target_profit')
        .pipe(reset_index, drop=True)
        .pipe(head))


    **Expression assignment**
    The parser also understands assignment in either R form (i.e. using <-) or Python (using '=').

    .. code-block::

        df <- pd.read_csv('inputs/test.csv')
        df =  pd.read_csv('inputs/test.csv')

    .. note::
        There are two other 'switches' that can parse alternative linking
        symbols, namely --r and --dot

    .. code-block::

        %%piper --r
        sample_sales() %>%
        group_by('location') %>%
        summarise(sales=('actual_sales', 'sum')) %>%
        assign(abc = lambda x: x.sales * 15.43,
               ratio = lambda x: ratio(x.sales, x.abc)) %>%
        relocate(['abc', 'ratio'], 'before', 'sales') %>%
        head(tablefmt='plain')

        location         abc    ratio    sales
        London      26210531        0  1698673
        Milan       31162265        0  2019589
        Paris       25582068        0  1657943

    .. code-block::

        %%piper --dot
        sample_sales()
        .. group_by('location')
        .. summarise(sales=('actual_sales', 'sum'))
        .. assign(abc = lambda x: x.sales * 15.43,
                  ratio = lambda x: ratio(x.sales, x.abc))
        .. relocate(['abc', 'ratio'], 'before', 'sales')
        .. head(tablefmt='plain')

    Parameters
    ----------
    text
        text to be parsed
    pipe_symbol
        default '>>' pipe character used to uniquely identify to the piper parse
        when to replace it with the .pipe() statement.
    info
        default False. If True - print rendered linked 'pipe' statement


    Returns
    -------
    revised_stmt
        revised 'piped' pandas pipeline statement
    execute
        'exec/eval' - for magic run_cmd to determine whether the python
        statement returned should executed or evaluated. if revised_stmt is
        'assigned' to another variable then 'exec' is used otherwise 'eval'

    '''
    if pipe_symbol in (None, False):
        pipe_symbol = '>>'

    # Check if text contains an R or Pythonic assignment symbol
    assignment, revised_text = pipe_assignment(text)

    # 1. Split into a list by newline
    text_as_list = revised_text.split('\n')
    text_as_list = [x.strip()+' ' for x in text_as_list if x != '']

    # 2. Remove commented out lines (I think its safer in case variables contain '#'
    text_as_list = [x for x in text_as_list if re.search('#', x) is None]

    # 3. Combine text back into one text then split by pipe_symbol
    revised_text = ''.join(text_as_list)
    text_as_list = revised_text.split(pipe_symbol)

    # 4. Parse function and split out parameters
    # That is, look for a '(' that starts a line, capturing the parameters
    # and subsequent ending bracket in the second match group.
    regex = r'(^.*?\()?(.*)?'
    search_lambda = lambda x: re.search(regex, x.strip()).groups() # type: ignore

    piped_list = [search_lambda(x) for x in text_as_list]

    # 5. Replace chosen pipe_symbol (e.g. >>) with '.pipe' chain function,
    # appending function + parameters
    piped_list = [pipe_function_parms(ix, x) for ix, x in enumerate(piped_list)] # type: ignore
    revised_stmt = ''.join(piped_list) # type: ignore

    text_as_list = revised_stmt.split('.pipe')
    f = lambda ix, x: '\n.pipe'+x if ix > 0 else x
    revised_stmt = '(' + ''.join([f(ix, x) for ix, x in enumerate(text_as_list)]) + ')'

    call_function = eval
    if assignment != '':
        call_function = exec
        revised_stmt = assignment + revised_stmt

    if info:
        print(revised_stmt)

    if debug:
        logger.info(f'call_function: {call_function}')
        logger.info(f'raw text:\n{revised_stmt}')

    return (revised_stmt, call_function)


# pipe_assignment() {{{1
def pipe_assignment(text: str) -> Tuple[str, str]:
    ''' Check if R or Pythonic assignment defined in text

    Parameters
    ----------
    text
        text string to be parsed

    Returns
    -------
    assignment
        assignment expression

    output_text
        remaining text expression(s)
    '''
    assignment = ''
    text_output = text

    r_assignment = r'(?<=\<\-)'
    match = re.search(fr'(.*){r_assignment}(.*)', text.strip(), flags=re.M | re.DOTALL)
    if match:
        assignment = match.groups()[0].strip().replace('<-', '= ')
        text_output = match.groups()[1].strip()

    if match is None:
        python_assignment = r'^(.*?\=)(.*)'
        match = re.match(python_assignment, text.strip(), flags=re.M | re.DOTALL)
        if match and '(' not in match.groups()[0]:
            assignment = match.groups()[0].strip()
            text_output = match.groups()[1].strip()

    return assignment, text_output


# pipe_function_parms() {{{1
def pipe_function_parms(idx: int, value: str) -> str:
    ''' pipe_parser() - join function with associated parameters

    Each statement within the pipeline is individually parsed into two sections:

        - the function name
        - the corresponding parameters

    Examples
    --------

    .. code-block::

        statements = [('head(', 'n=10)')]
        result = [pipe_function_parms(idx, x) for idx, x in enumerate(statements)]
        result
        > ['head(n=10)']

    Parameters
    ----------
    idx
        list index currently being processed
    value
        function + optional parameters required to form .pipe() statement

    Returns
    -------
    piped functional statement
    '''
    if value[0] is None:
        return value[1]

    if idx == 0:
        return f"{''.join(value)}"

    if value[1] == ')':
        return f".pipe({value[0][:-1]}{''.join(value[1:])}"

    return f".pipe({value[0][:-1]}, {''.join(value[1:])}"


# paste() {{{1
def paste(source='text'):
    ''' Convert a clipboard sourced list from text or excel file

    Examples
    --------
    columns = paste(source='text')

    Parameters
    ----------
    source
        Default 'vertical'
        vertical        - copy from a vertical list of values (usually Excel)
        horizontal      - copy from a horizontal list of values (usually Excel)
        horizontal_list - return a horizontal list

    Returns
    -------
    Clipboard contents
    '''
    if source == 'vertical':
        dx = pd.read_clipboard(header=None, names=['x'])
        data = "['" + "', '".join(dx.x.values.tolist()) + "']"

    if source == 'horizontal':
        dx = pd.read_clipboard(sep='\t')
        data = dx.columns.tolist()

    if source == 'horizontal_list':
        dx = pd.read_clipboard(sep='\t')
        data = "['" + "', '".join(dx.columns.tolist()) + "']"

    return data


# set_sql_file {{{1
def set_sql_file(from_file=None, to_file=None, template_values=True):
    ''' Set SQL file text to 'template' substitutional values

    Examples
    --------

    Set/replace existing SQL text file to template format:

    .. code-block::

        project_dir = Path.home() / 'documents'

        from_file = project_dir / 'Scratchpad.sql'
        to_file = project_dir / 'Scratchpad_revised.sql'
        set_sql_file(from_file, to_file, template_values=True)


    To convert a 'templated' sql text file to use hardcoded schema values:

    .. code-block::

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
        True (convert sql text to use 'template' variables) or,
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
        True (convert sql text to use 'template' variables) or,
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


# _file_with_ext() {{{1
def _file_with_ext(filename, extension='.xlsx'):
    ''' For given filename, check if required extension present.

        If not, append and return.
    '''
    if re.search(extension+'$', filename) is None:
        return filename + extension

    return filename


# _get_qual_file() {{{1
def _get_qual_file(folder, file_name, ts_prefix=True):
    ''' Get qualified xl file name

    Examples
    --------

    .. code-block::

        _get_qual_file(folder, file_name, ts_prefix=False)
        _get_qual_file(folder, file_name, ts_prefix='date')
        _get_qual_file(folder, file_name, ts_prefix='time')

    Parameters
    ----------
    folder
        folder path
    file_name
        file name to create.
    ts_prefix
        default False (no timestamp)
        True (timestamp prefix)
        'date' (date only)

    Returns
    -------
    qual_file
        qualified file name
    '''
    if ts_prefix == 'date':
        ts = "{:%Y%m%d_}".format(datetime.now())
        qual_file = normpath(join(folder, ts + file_name))
    elif ts_prefix:
        ts = "{:%Y%m%d_%H%M}_".format(datetime.now())
        qual_file = normpath(join(folder, ts + file_name))
    else:
        qual_file = normpath(join(folder, file_name))

    return qual_file


# _get_sql_text {{{1
def _get_sql_text(sql, variables, debug=False):
    ''' Get SQL text and substitute variables

    Parameters
    ----------
    sql
        sql text
    variables
        dictionary of variables to be used for replacing substitutional
        variables.
    debug
        Default False. If True, provide logging/debug information

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
