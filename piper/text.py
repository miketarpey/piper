import re
import logging


logger = logging.getLogger(__name__)


# pipe_parser() {{{1
def pipe_parser(text, pipe_symbol=None, info=False, debug=False):
    ''' piper parser - for given text, split by pipe_symbol then for each
    split statement, wrap with .pipe().

    Return completed pipeline of functions/commands, plus a boolean to determine
    whether the statement should be executed or not.

    Parameters
    ----------
    text: str - text to be parsed

    pipe_symbol: str - default '>>' pipe character used to 'link' statement chain
                     together

    info: boolean - default False. If True - print rendered linked 'pipe' statement

    Returns
    -------
    revised_stmt: str - revised 'piped' pandas pipeline statement

    execute: function - 'exec/eval' - for magic run_cmd to determine whether the
                        python statement returned should executed or evaluated.
                        if revised_stmt is 'assigned' to another variable
                        then 'exec' is used otherwise 'eval'

    Example
    -------
    cell_value, call_function = pipe_parser(cell, pipe_symbol=pipe_symbol, info=info, debug=False)
    '''
    if pipe_symbol in (None, False):
        pipe_symbol = '>>'

    # Take a copy of the input text
    revised_text = text

    # 0. Check to see if an assignment has been made
    # Example:     df <- pd.read_csv('inputs/test.csv')
    # Converts to: df =  pd.read_csv('inputs/test.csv')
    r_assignment = r'(?<=\<\-)'
    match = re.search(fr'(.*){r_assignment}(.*)', text.strip(), flags=re.M | re.DOTALL)
    if match:
        assignment = match.groups()[0].strip().replace('<-', '= ')
        revised_text = match.groups()[1].strip()

    if match is None:
        python_assignment = r'^(.*?\=)(.*)'
        match = re.match(python_assignment, text.strip(), flags=re.M | re.DOTALL)
        if match and '(' not in match.groups()[0]:
            assignment = match.groups()[0].strip()
            revised_text = match.groups()[1].strip()
        else:
            # Ensure no 'assignment takes place
            match = None

    # 1. Split by newline
    revised_text = revised_text.split('\n')
    revised_text = [x.strip()+' ' for x in revised_text if x != '']

    # 2. Remove commented out lines (I think its safer in case variables contain '#'
    revised_text = [x for x in revised_text if re.search('#', x) is None]

    # 3. Combine text back into one text then split by pipe_symbol
    revised_text = ''.join(revised_text).split(pipe_symbol)

    # 4. Parse function and split out parameters
    # That is, look for a '(' that starts a line, capturing the parameters
    # and subsequent ending bracket in the second match group.
    regex = r'(^.*?\()?(.*)?'
    revised_text = [re.search(regex, x.strip()).groups() for x in revised_text]

    # 5. Replace chosen pipe_symbol (e.g. >>) with '.pipe' chain function,
    # appending function + parameters
    revised_stmt = [join_function_parms(idx, x) for idx, x in enumerate(revised_text)]
    revised_stmt = ''.join(revised_stmt)

    revised_stmt = revised_stmt.split('.pipe')
    f = lambda ix, x: '\n.pipe'+x if ix > 0 else x
    revised_stmt = '(' + ''.join([f(ix, x) for ix, x in enumerate(revised_stmt)]) + ')'

    call_function = eval
    if match:
        call_function = exec
        revised_stmt = assignment + revised_stmt

    if info:
        print(revised_stmt)

    if debug:
        logger.info(f'call_function: {call_function}')
        logger.info(f'raw text:\n{revised_stmt}')

    return (revised_stmt, call_function)


def join_function_parms(idx, value):
    ''' join function with associated parameter values

    Parameters
    ----------
    idx: (int) - list index currently being processed

    value: (str) - function plus optional parameters to be joined
                   to form .pipe() statement

    Example
    -------
    statements = [('head(', 'n=10)')]
    result = [join_function_parms(idx, x) for idx, x in enumerate(statements)]
    result
    > ['head(n=10)']
    '''
    if value[0] is None:
        return value[1]

    if idx == 0:
        return f"{''.join(value)}"

    if value[1] == ')':
        return f".pipe({value[0][:-1]}{''.join(value[1:])}"

    return f".pipe({value[0][:-1]}, {''.join(value[1:])}"
