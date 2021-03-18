import numpy as np #type: ignore
import pandas as pd #type: ignore
import logging
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ratio - calculate ratio between 2 values or pd.Series {{{1
def ratio(value1: Union[int, float, pd.Series],
          value2: Union[int, float, pd.Series],
          precision: int = 2,
          percent: bool = False,
          format: bool = False) -> Any:
    ''' ratio - ratio/percentage of two values (or pd.Series)

    Provide parameters to adjust decimal rounding and a 'format'
    option to send back string rather than numeric with % appended.
    e.g. '33.43%'

    Passes back np.inf value for 'divide by zero' use case.

    Usage example
    -------------
    s1 = pd.Series([10, 20, 30])
    s2 = pd.Series([1.3, 5.4, 3])
    ratio(s1, s2)

    |    |     0 |
    |---:|------:|
    |  0 |  7.69 |
    |  1 |  3.70 |
    |  2 | 10.00 |


    Parameters
    ----------
    value1: integer, float, or pd.Series

    value2: integer, float, or pd.Series


    Returns
    -------
    float - if values1 and 2 are single int/float values
    pd.Series - if values1 and 2 are pd.Series

    '''
    if isinstance(value1, pd.Series):
        if percent:
            result = (value1 * 100 / value2)
        else:
            result = (value1 / value2)

        if precision is not None:
            result = result.round(precision)

        if format:
            result = result.astype(str) + '%'

        return result

    # Assumption, if not pd.Series, we are dealing
    # with individual int or float values
    try:
        if percent:
            result = ((value1 * 100)/ value2)
        else:
            result = value1 / value2

        if precision is not None:
            result = round(result, precision)

        if format:
            result = f'{result}%'

    except ZeroDivisionError as e:
        logger.info(e)
        return np.inf

    return result
