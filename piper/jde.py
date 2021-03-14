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
                  batch: str = 'ABC',
                  start: int = 100,
                  step: int = 100) -> pd.DataFrame:
    ''' Add 'standard' JDE timestamp/default columns.
    For given dataframe, adds the following standard Z-file columns.

    User ID (edus)
    Batch Number (edbt)
    Transaction Number (edtn)
    Line Number (edln)


    Parameters
    ----------
    df : the pandas dataframe object

    col_prefix : 2 character (e.g. 'ed') column name prefix to be
        applied to the added columns

    userid : default userid text value

    batch : 2 character prefix to concatenated to current timestamp

    trans_no : start number in xxln column

    step : step increment in xxln column


    Returns
    -------
    A pandas dataframe
    '''
    timestamp = datetime.now().strftime('_%Y%m%d')
    start_position = 0

    range_seq = range(start, (df.shape[0]+1)*step, step)

    df.insert(start_position, f'{col_prefix}us', userid)
    df.insert(start_position+1, f'{col_prefix}bt', batch + timestamp)
    df.insert(start_position+2, f'{col_prefix}tn', 1)

    df.insert(start_position+3, f'{col_prefix}ln', pd.Series(range_seq))

    return df
