from IPython.core.magic import line_magic, cell_magic, line_cell_magic, Magics, magics_class
from IPython.core import magic_arguments
from datetime import datetime
from piper.version import __version__
import logging
import re


# KEEP:: In case of issues where you need to trace the function
log_fmt = '%(asctime)s %(name)-10s %(levelname)-4s %(funcName)-14s %(message)s'

# This log format shows 'minimal' information
log_fmt = '%(message)s'

# To suppress logging - uncomment line below and comment the one after :)
# logging.basicConfig(level=logging.ERROR, format=log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
logging.basicConfig(level=logging.INFO, format=log_fmt, datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

runtime = datetime.now().strftime('%A, %d %B %Y %H:%M:%S')
logger.info(f"piper version {__version__}, last run: {runtime}")


# run_cmd {{{1
def run_cmd(execute, cell_value, env_variables=None):
    '''

    '''
    if env_variables is None:
        env_variables = get_ipython().user_ns

    try:
        if execute:
            return exec(cell_value, env_variables)

        return eval(cell_value, env_variables)

    except SyntaxError as e:
        logger.info(f':: NOTE :: use %piper/%%piper --info for more information')
        logger.info(f':: line number :: {e.lineno}')
        logger.info(f':: error text  :: {e.text}')
    except AttributeError as e:
        logger.info(f':: NOTE :: use %piper/%%piper --info for more information')
        logger.info(e)


# PipeMagic {{{1
@magics_class
class PipeMagic(Magics):

    @line_cell_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('--info', '-i', action='store_true',
                              help='Print the equivalent pipe statements')
    @magic_arguments.argument('--rdelimitter', '-r', action='store_true',
                              help='delimitter to be used')
    def piper(self, line='', cell=None):

        if cell is not None and cell.strip() != '':
            args = magic_arguments.parse_argstring(self.piper, line)
            logger.debug(args)

            info = False
            if args.info: info = True

            delimitter = False
            if args.rdelimitter: delimitter = '%>%'

            cell_value, execute = parse_text(cell, delimitter=delimitter, info=info, debug=False)
            return run_cmd(execute, cell_value)

        else:

            if line.strip() != '':
                line_value, execute = parse_text(line, info=False, debug=False)
                return run_cmd(execute, line_value)

            return line


# parse_text() {{{1
def parse_text(text, delimitter=None, comment='#', info=False, debug=False):

    def pipe_functions(idx, value):
        ''' break/split function name from parms into
        a series of chained pipe expressions
        '''
        if value[0] is None:
            return value[1]

        if idx == 0:
            return f"{''.join(value)}"

        if value[1] == ')':
            return f".pipe({value[0][:-1]}{''.join(value[1:])}"

        return f".pipe({value[0][:-1]}, {''.join(value[1:])}"

    if delimitter in (None, False):
        delimitter = '>>'

    # 0. Check to see if an assignment has been made
    # E.g. df <- pd.read_csv('inputs/test.csv')
    match = re.search('(.*)(?<=\<\-)(.*)', text, flags=re.M | re.DOTALL)
    if match:
        assignment = match.groups()[0].strip().replace('<-', '= ')
        text = match.groups()[1]

    # 1. Split by newline
    revised_text = text.split('\n')
    revised_text = [x.strip() for x in revised_text if x != '']
    if debug:
        logger.info(revised_text)

    # 2. Remove commented out lines
    revised_text = [x.strip() for x in revised_text if re.search(comment, x) is None]
    if debug:
        logger.info(revised_text)

    # 3. Select lines with delimitter
    regex = f'{delimitter}$'
    revised = [x.strip() for x in revised_text if re.search(regex, x.strip()) is not None]
    revised.append(revised_text[-1])
    if debug:
        logger.info(revised)

    # 4. Combine again into one text then split by delimitter
    revised_text = ''.join(revised_text)
    revised_text = revised_text.split(delimitter)

    # 5. Parse function and split out parameters
    regex = '(^.*?\()?(.*)?'
    revised_text = [re.search(regex, x.strip()).groups() for x in revised_text]
    if debug:
        logger.info(revised_text)

    # 6. Replace chosen delimitter (e.g. >>) with '.pipe' chain function,
    # appending function + parameters
    revised_stmt = [pipe_functions(idx, x) for idx, x in enumerate(revised_text)]
    revised_stmt = ''.join(revised_stmt)

    revised_stmt = revised_stmt.split('.pipe')
    f = lambda ix, x: '\n.pipe'+x if ix > 0 else x
    revised_stmt = '(' + ''.join([f(ix, x) for ix, x in enumerate(revised_stmt)]) + ')'

    if match:
        revised_stmt = assignment + revised_stmt
        if info:
            print(revised_stmt)

        return (revised_stmt, True)

    if info:
        print(revised_stmt)

    return (revised_stmt, False)


ip = get_ipython()
ip.register_magics(PipeMagic)
