import logging
import pandas as pd
from datetime import datetime

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


# add_jde_batch() {{{1
def add_jde_batch(df: pd.DataFrame,
                  col_prefix: str = 'ed',
                  userid: str = 'userid',
                  batch_prefix: str = 'ABC',
                  start: int = 100,
                  step: int = 100,
                  inplace: bool = False) -> pd.DataFrame:
    ''' Add 'standard' JDE timestamp/default column and values to
    a dataframe - allowing it to be used (for example) for Z-file
    upload.

    Note: returns a copy of the given dataframe


    Parameters
    ----------
    df : the pandas dataframe object

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
