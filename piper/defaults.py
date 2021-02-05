import sys
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# IPython setup
pd.set_option("display.max_columns", 80)
pd.set_option('max_colwidth', 120)

# Seaborn for plotting and styling
sns.set(color_codes=True)

# useful variables
current_date = datetime.now().strftime('%Y%m%d')
