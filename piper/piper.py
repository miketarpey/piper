from IPython import get_ipython
from piper.magics import PipeMagic

# Register magics
ip = get_ipython()
ip.register_magics(PipeMagic)

from piper.verbs import *
from piper.factory import *
