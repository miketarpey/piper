import sys
import logging
from datetime import datetime
from pathlib import Path
from piper.version import __version__

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# IPython setup
pd.set_option("display.max_columns", 80)
pd.set_option('max_colwidth', 120)

# Seaborn for plotting and styling
sns.set(color_codes=True)

# useful variables
current_date = datetime.now().strftime('%Y%m%d')
