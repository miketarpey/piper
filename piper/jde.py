import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


# add_jde_batch() {{{1
def add_jde_batch(df, col_prefix='ed', userid='userid',
                  batch_prefix='ABC', start=100, step=100, inplace=False):
    ''' Add 'standard' JDE timestamp/default column and values to
    a dataframe - allowing it to be used (for example) for Z-file
    upload.

    Note: returns a copy of the given dataframe


    Parameters
    ----------
    df : pandas dataframe

    col_prefix : 2 character (e.g. 'ed') column name prefix

    userid : default userid text value

    batch_prefix : prefix used in xxbt column

    start : start number in xxln column

    step : step increment in xxln column


    Returns
    -------
    pandas dataframe (copy)
    '''
    timestamp = datetime.now().strftime('_%Y%m%d')
    start_position = 0

    range_seq = range(start, (df.shape[0]+1)*step, step)
    line_sequence = pd.Series(range_seq)

    df.insert(start_position, f'{col_prefix}us', userid)
    df.insert(start_position+1, f'{col_prefix}bt', batch_prefix + timestamp)
    df.insert(start_position+2, f'{col_prefix}tn', 1)
    df.insert(start_position+3, f'{col_prefix}ln', line_sequence)

    return df
