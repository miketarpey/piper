import logging
import pandas as pd
from functools import wraps

logger = logging.getLogger(__name__)


# shape() {{{1
def shape(*args2, **kwargs2):
    '''
    shape decorator allows reporting of shape (rows, cols) info
    when a pd.DataFrame object is returned from decorated function.

    Parameters
    ----------
    debug : False (default), True show keyword pair arguments
                             and arguments (*kwargs, *args)

    Example:

    @shape(debug=False)
    def read_tsv(*args, **kwargs):

        if kwargs == {}:
            df = pd.read_csv(*args, sep='\t')
        else:
            kwargs.update({'sep':'\t'})
            df = pd.read_csv(*args, **kwargs)

        return df

    read_tsv('inputs/20160321_01_A940_(cleaned).tsv').head(2)
    INFO:__main__:rows, columns: (8620, 11)

    '''
    def outer_function(func):

        @wraps(func)
        def shape_deco(*args, **kwargs):

            if kwargs2 != {}:
                if kwargs2['debug']:
                    logger.info(f'Arguments for {args}')
                    logger.info(f'Arguments for {kwargs}')

            result = func(*args, **kwargs)

            if isinstance(result, pd.Series):
                if kwargs2['debug']:
                    logger.info(f'{func.__name__} -> {result.shape[0]} rows')
                logger.info(f'{result.shape[0]} rows')

            if isinstance(result, pd.DataFrame):
                if kwargs2['debug']:
                    logger.info(f'{func.__name__} -> {result.shape[0]} rows, {result.shape[1]} columns')
                logger.info(f'{result.shape[0]} rows, {result.shape[1]} columns')

            return result

        return shape_deco

    return outer_function
