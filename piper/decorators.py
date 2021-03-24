import logging
import pandas as pd
from functools import wraps

logger = logging.getLogger(__name__)


# shape() {{{1
def shape(trigger='after', keyword='shape', debug=False):
    ''' Shape decorator - Experimental

    Shape decorator allows reporting of shape \(rows, cols\) info within a
    decorated function.

    If the decorated function has a keyword 'shape' specified then the shape
    functionality is triggered. The shape function can be invoked before or
    after the decorated function using the trigger parameter.


    Parameters
    ----------
    trigger
        trigger action before or after decorated function invoked.
    keyword
        default 'shape'. keyword to trigger the decorated action/event.
    debug
        default False. True shows decorated arguments and keyword args.
    '''
    def wrapper_function(func):

        @wraps(func)
        def decorated_function(*args, **kwargs):

            if debug:
                logger.info(f'args: {args}')
                logger.info(f'kwargs: {kwargs}')

            if trigger == 'before':
                if kwargs.get(keyword):
                    _shape(args[0])

            result = func(*args, **kwargs)

            if trigger == 'after':
                if kwargs.get(keyword):
                    _shape(result)

            return result

        return decorated_function

    return wrapper_function


def _shape(df):
    ''' show df.shape in logger output '''

    if isinstance(df, pd.Series):
        logger.info(f'{df.shape[0]} rows')

    if isinstance(df, pd.DataFrame):
        logger.info(f'{df.shape[0]} rows, {df.shape[1]} columns')
