from IPython.core.magic import line_magic, cell_magic, line_cell_magic, Magics, magics_class
from IPython.core import magic_arguments
from datetime import datetime
from piper.text import pipe_parser
from piper.version import __version__
import logging


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
        result = execute(cell_value, env_variables)
        return result
    except Exception as e:
        logger.info(f'Use %piper/%%piper --info to see rendered pandas pipe statement')
        logger.info(e)


# PipeMagic {{{1
@magics_class
class PipeMagic(Magics):

    @line_cell_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('--info', '-i', action='store_true', help='Print the equivalent pipe statements')
    @magic_arguments.argument('--debug', '-d', action='store_true', help='Debug information')
    @magic_arguments.argument('--r', '-r', action='store_true', help='pipe_symbol to be used')
    def piper(self, line='', cell=None):

        info, debug = False, False
        pipe_symbol = '>>'

        if cell is not None and cell.strip() != '':

            args = magic_arguments.parse_argstring(self.piper, line)
            logger.debug(args)

            if args.info:
                info = True

            if args.debug:
                debug = True

            if args.r:
                pipe_symbol = '%>%'

            cell_value, execute = pipe_parser(cell, pipe_symbol=pipe_symbol, info=info, debug=debug)
            return run_cmd(execute, cell_value)
        else:
            if line.strip() != '':
                line_value, execute = pipe_parser(line, pipe_symbol=pipe_symbol, info=info, debug=debug)
                return run_cmd(execute, line_value)

            return line

ip = get_ipython()
ip.register_magics(PipeMagic)
