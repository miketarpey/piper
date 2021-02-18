'''
Styler function style summary
=============================
Styler.applymap(func)         : element-wise styles
Styler.apply(func, axis=0)    : column-wise styles
Styler.apply(func, axis=1)    : row-wise styles
Styler.apply(func, axis=None) : tablewise styles
'''
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

def get_default_style():

    style = [{'selector': '*',
            'props': [('border', '1px solid #9EB6CE'),
                      ('border-width', '1px 1px 1px 1px'),
                      ('font-size', '15px'),
                      ('font-weight', 200),
                      ('table-layout', 'auto'),
                      ('border-spacing', '0px'),
                      ('border-collapse', 'collapse')
            ]},
           {'selector': 'tbody tr:hover td',
            'props': [('background-color', '#DCDCDC !important'),
                      ('color', '#000'),
            ]},
           {'selector': 'th',
            'props': [('background-color', '#E4ECF7'),
                      ('font-weight', 600),
                      ('font-size', '15px'),
                      ('border', '1px solid #9EB6CE'),
                      ('border-width', '0px 1px 1px 0px')
            ]},
           {'selector': 'td',
            'props': [('border', '0px'),
                      ('font-size', '14px'),
                      ('padding', '2px 5px 2px 5px'),
                      ('border', '1px solid #9EB6CE'),
                      ('border-width', '1px 1px 1px 1px'),
            ]},
            ]

    return style


def highlight_values(s, criteria=None, type=None, color='red'):
    '''
    For given Series, highlight individual values using a
    specific (equal) value or list of values

    '''
    logger.debug(f'criteria: {criteria}')

    if type is None:
        color = f'color: {color}'

    if type == 'background':
        color = f'background-color: {color}'

    if isinstance(criteria, list):
        style = [color if v in criteria else '' for v in s]
    else:
        style = [color if v == criteria else '' for v in s]

    logger.debug(style)

    return style


def highlight_min(s, type=None, color='yellow'):
    '''
    For given Series, highlight min values
    '''
    s2 = s == s.min()
    logger.debug(f'criteria: {s2}')

    if type is None:
        color = f'color: {color}'

    if type == 'background':
        color = f'background-color: {color}'

    style = [color if v else '' for v in s2]
    logger.debug(style)

    return style


def highlight_max(s, type=None, color='yellow'):
    '''
    For given Series, highlight max values
    '''
    s2 = s == s.max()
    logger.debug(f'criteria: {s2}')

    if type is None:
        color = f'color: {color}'

    if type == 'background':
        color = f'background-color: {color}'

    style = [color if v else '' for v in s2]
    logger.debug(style)

    return style


def highlight_null_rows(s, column=None, operator='=', type=None, color='blue'):
    '''
    '''
    length = s.shape[0]
    logger.debug(s)

    if operator == '=':
        compare_value = np.isnan(s[column]).any()

    if operator == '!=':
        compare_value = ~np.isnan(s[column]).any()

    logger.debug(compare_value)

    if type is None:
        color = f'color: {color}'

    if type == 'background':
        color = f'background-color: {color}'

    if compare_value:
        return [f'{color}'] * length
    else:
        return [''] * length


def highlight_rows(s, column='B', operator='=', criteria=0, type=None, color='blue'):
    '''
    df.style.apply(highlight_rows, axis=1)

    Example:
    ========

    import pandas as pd
    import numpy as np

    np.random.seed(24)
    df = pd.DataFrame({'A': np.linspace(1, 10, 10)})
    df2 = pd.DataFrame(np.random.randn(10, 4), columns=list('BCDE'))

    df = pd.concat([df, df2], axis=1)
    df.iloc[0, 2] = np.nan

    (df.style.hide_index()
             .apply(highlight_rows, column='C', operator='<=', criteria=1, axis=1)
             .apply(highlight_rows, column='C', operator='<=', criteria=1, type='background', color='yellow', axis=1)
    )

    '''
    length = s.shape[0]
    logger.debug(s)

    if operator == '=':
        compare_value = s[column] == criteria

    if operator == '<':
        compare_value = s[column] < criteria

    if operator == '<=':
        compare_value = s[column] <= criteria

    if operator == '>':
        compare_value = s[column] > criteria

    if operator == '>=':
        compare_value = s[column] >= criteria

    logger.debug(compare_value)

    if type is None:
        color = f'color: {color}'

    if type == 'background':
        color = f'background-color: {color}'

    if compare_value:
        return [f'{color}'] * length
    else:
        return [''] * length
