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


def boolean_filter(s, operator=None, value=0):
    ''' For series or value, evaluate if True/False against
    operator and value and return result

    Example
    -------
    boolean_filter(10, '>', 0)
    > True

    boolean_filter(xx['isna'], '>', 0)
    > True


    Parameters
    ----------
    s
        value(int, float, str, pd.Series)
        value(s) to be evaluated (True/False) against operator
    operator
        str - default '=' , possible values:
       '!=', '<', '<=', '>', '>=', 'min', 'max', 'null'
    value
        str/integer - value to compare against


    Returns
    -------
    ndArray of evaluated boolean values.

    '''
    if operator is None:
        operator = '='

    logger.debug(f's -> {type(s)}')
    logger.debug(f'operator -> {operator}')
    logger.debug(f'value -> {type(value)}')

    if isinstance(s, pd.Series) and s.dtype == 'O' and\
        operator in ('min', 'max'):
            raise ValueError('min and max numeric values only')

    # If 1st and 2nd comparators are not Series,
    # Then use simple eval comparison
    if not isinstance(s, pd.Series) and \
        not isinstance(value, pd.Series):

        # eval_selection = {'=': s == value, '!=': s != value}
        # if operator in ('<=', '>=', 'max', 'null', 'min'):
        #     error_msg = f'String not comparable with Series using {operator}'
        #     raise NameError(error_msg)

        eval_selection = {
            'null': pd.isnull(s),
            '!null': ~pd.isnull(s),
            '=': s == value,
            '!=': s != value,
            '<': s < value,
            '<=': s <= value,
            '>': s > value,
            '>=': s >= value}

        result = eval_selection.get(operator)
        logger.debug(result)

        return result


    if isinstance(s, pd.Series):
        eval_selection = {
            'min': s == s.min(),
            'max': s == s.max(),
            'null': np.isnan(s),
            '!null': ~np.isnan(s),
            '=': s == value,
            '!=': s != value,
            '<': s < value,
            '<=': s <= value,
            '>': s > value,
            '>=': s >= value
        }

        result = eval_selection.get(operator)
        logger.debug(result)

        return result


def highlight_min(s, css='color: blue; background-color: lightgrey'):
    ''' For given Series, highlight min values '''

    result = boolean_filter(s, operator='min', value=None)
    style = [css if v else '' for v in result]

    return style


def highlight_max(s, css='color: red; background-color: lightgrey'):
    ''' For given Series, highlight max values '''

    result = boolean_filter(s, operator='max', value=None)
    style = [css if v else '' for v in result]

    return style


def highlight_values(s, values=None, css=None):
    '''
    For given Series, highlight individual values using a
    specific (equal) value or list of values

    '''
    if css is None:
        css = 'color: red'

    if isinstance(values, list):
        style = [css if v in values else '' for v in s]
    else:
        style = [css if v == values else '' for v in s]

    return style


def highlight_rows(row, column=None, operator='=', value=0, css=None):
    '''

    Example:
    ========
    import pandas as pd
    import numpy as np

    np.random.seed(24)
    df = pd.DataFrame({'A': np.linspace(1, 10, 10)})
    df2 = pd.DataFrame(np.random.randn(10, 4), columns=list('BCDE'))

    df = pd.concat([df, df2], axis=1)
    df.iloc[0, 2] = np.nan

    df.style.apply(highlight_rows, axis=1)
    '''
    if column is None:
        raise KeyError(f'column {column} not found!')

    row_length = row.shape[0]

    if css is None:
        css = 'color: blue'

    result = boolean_filter(row[column], operator=operator, value=value)

    row_css = [''] * row_length
    if np.any(result):
        row_css = [css] * row_length

    return row_css


def get_style(style='default'):
    '''
    '''
    if style == 'default':
        return _style_excel()

    if style == 'ibm':
        return _style_ibmi()


def _style_excel():
    ''' Retrieve default css format

    '''
    style = [{'selector': '*',

              'props': [('border', '1px solid #9EB6CE'),
                      ('border-width', '1px 1px 1px 1px'),
                      ('font-size', '12px'),
                      ('font-weight', 200),
                      ('table-layout', 'auto'),
                      ('padding', '8px 8px 8px 8px'),
                      ('border-spacing', '0px'),
                      ('border-collapse', 'collapse')
            ]},
             {'selector': 'caption',
                'props': [('color', 'black'),
                          ('font-size', '15px')
            ]},
           {'selector': 'tbody tr:hover td',
            'props': [('background-color', '#DCDCDC !important'),
                      ('color', '#000'),
            ]},
           {'selector': 'th',
            'props': [('background-color', '#E4ECF7'),
                      ('font-weight', 600),
                      ('font-size', '13px'),
                      ('border', '1px solid #9EB6CE'),
                      ('border-width', '0px 1px 1px 0px')
            ]},
           {'selector': 'td',
            'props': [('border', '0px'),
                      ('font-size', '12px'),
                      ('border', '1px solid #9EB6CE'),
                      ('border-width', '1px 1px 1px 1px'),
            ]},
            ]

    return style


def _style_ibmi():
    ''' Retrieve ibmi format/properties

    (merge_df.style.set_properties(**get_style(style='ibm')))
    '''

    style = {'background-color': 'black',
              'color': 'lightgreen',
              'font-size': '140%'}

    return style
