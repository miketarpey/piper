import re
import logging
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
    ```python
    %%piper --info
    get_sample_sales() >>
    select('-target_profit') >>
    reset_index(drop=True) >>
    head()
    ```

    is rendered by the parser as:

    ```python
    %%piper --info
    get_sample_sales() >>
    select('-target_profit') >>
    reset_index(drop=True) >>
    head()
    ```

    **Expression assignment**
    The parser also understands assignment in either R form (i.e. using <-) or Python (using '=').
    ```python
    df <- pd.read_csv('inputs/test.csv')
    df =  pd.read_csv('inputs/test.csv')
    ```

    Parameters
    ----------
    text: text to be parsed

    pipe_symbol: default '>>' pipe character used to uniquely identify to
        the piper parse when to replace it with the .pipe() statement.

    info: default False. If True - print rendered linked 'pipe' statement


    Returns
    -------
    revised_stmt: revised 'piped' pandas pipeline statement

    execute: 'exec/eval' - for magic run_cmd to determine whether the
             python statement returned should executed or evaluated.
             if revised_stmt is 'assigned' to another variable
             then 'exec' is used otherwise 'eval'

    '''
    if pipe_symbol in (None, False):
        pipe_symbol = '>>'

    # Check if text contains an R or Pythonic assignment symbol
    assignment, revised_text = check_assignment(text)

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
    piped_list = [join_function_parms(ix, x) for ix, x in enumerate(piped_list)] # type: ignore
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


def check_assignment(text: str) -> Tuple[str, str]:
    ''' Check if R or Pythonic assignment defined in text


    Parameters
    ----------
    text - text string to be parsed


    Returns
    -------
    assignment - assignment expression

    output_text - remaining text expression(s)

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


def join_function_parms(idx: int, value: str) -> str:
    ''' pipe_parser() - join function with associated parameters

    Each statement within the pipeline is individually parsed into two sections:
    - the function name
    - the corresponding parameters


    Examples
    --------
    statements = [('head(', 'n=10)')]
    result = [join_function_parms(idx, x) for idx, x in enumerate(statements)]
    result
    > ['head(n=10)']


    Parameters
    ----------
    idx: list index currently being processed

    value: function + optional parameters required to form .pipe() statement


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
