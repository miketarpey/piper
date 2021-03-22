from IPython.core.magic import line_cell_magic, Magics, magics_class
from IPython.core import magic_arguments
from piper.text import pipe_parser
import logging

logger = logging.getLogger(__name__)


# PipeMagic {{{1
@magics_class
class PipeMagic(Magics):

    @line_cell_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('--info', '-i', action='store_true', help='Print the equivalent pipe statements')
    @magic_arguments.argument('--debug', '-d', action='store_true', help='Debug information')
    @magic_arguments.argument('--r', '-r', action='store_true', help='pipe_symbol magrittr R version')
    @magic_arguments.argument('--dot', '-e', action='store_true', help='pipe_symbol ".."')
    def piper(self, line='', cell=None):
        ''' Piper line and cell magic '''

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

            if args.dot:
                pipe_symbol = '..'

            cell_value, execute = pipe_parser(cell, pipe_symbol=pipe_symbol, info=info, debug=debug)
            return self.run_cmd(execute, cell_value)
        else:
            if line.strip() != '':
                line_value, execute = pipe_parser(line, pipe_symbol=pipe_symbol, info=info, debug=debug)
                return self.run_cmd(execute, line_value)

            return line


    # run_cmd {{{1
    def run_cmd(self, execute, cell_value, env_variables=None):
        ''' Execute or evaluate the line or cell contents '''

        if env_variables is None:
            env_variables = get_ipython().user_ns

        try:
            result = execute(cell_value, env_variables)
            return result
        except Exception as e:
            logger.info(f'Use %piper/%%piper --info to see rendered pandas pipe statement')
            logger.info(e)

