import re
import pandas as pd
import numpy as np
import logging
from functools import wraps
from piper.io import _file_with_ext
from piper.decorators import shape
from pandas.core.common import flatten
from pandas.api.types import is_period_dtype
from pandas.api.types import is_datetime64_any_dtype
from collections import namedtuple
from datetime import datetime
from piper.xl import WorkBook
import logging

logger = logging.getLogger(__name__)


# head() {{{1
def head(df, n=4, info=True):
    ''' head: R like function, easier to use and control (than say df.head())

    Example:
    --------
    head(df)


    Parameters
    ----------
    df : dataframe

    n : number of rows to display, default n=2


    Returns
    -------
    returns input dataframe using .head() pandas function
    '''
    if info:
        if isinstance(df, pd.Series):
            logger.info(f'{df.shape[0]} rows')

        if isinstance(df, pd.DataFrame):
            logger.info(f'{df.shape[0]} rows, {df.shape[1]} columns')

    return df.head(n=n)


# tail() {{{1
def tail(df, n=4, info=True):
    ''' tail: R like function, easier to use and control (than say df.tail())

    Example:
    --------
    tail(df)


    Parameters
    ----------
    df : dataframe

    n : number of rows to display, default n=2


    Returns
    -------
    returns input dataframe using .tail() pandas function
    '''
    if info:
        if isinstance(df, pd.Series):
            logger.info(f'{df.shape[0]} rows')

        if isinstance(df, pd.DataFrame):
            logger.info(f'{df.shape[0]} rows, {df.shape[1]} columns')

    return df.tail(n=n)


# info() {{{1
def info(df, n_dupes=False, fillna=False, memory_info=True):
    ''' Alternative to df.info(), show all columns that have invalid data

    Parameters
    ----------
    df : dataframe

    n_dupes: boolean - column showing the numbers of groups of duplicate records
             default: False

    fillna: boolean - filter out only columns which contain nulls/np.nan
            default: False

    memory: boolean - If True, show memory returned dataframe consumption

    '''
    null_list = [df[col].isnull().sum() for col in df]
    na_list = [df[col].isna().sum() for col in df]
    dtypes_list = [df[col].dtype for col in df]
    nunique_list = [df[col].nunique() for col in df]
    dupes_list = [df.duplicated(col, keep=False).sum() for col in df]
    total_list = [df[col].value_counts(dropna=False).sum() for col in df]

    dicta = {'columns': df.columns.values, 'type': dtypes_list,
             'n': total_list, 'isna': na_list, 'isnull': null_list,
             'unique': nunique_list}

    if n_dupes:
        dicta.update({'n dupes': dupes_list})

    dr = pd.DataFrame.from_dict(dicta)

    if fillna:
        return dr.query('isna > 0')

    if memory_info:
        memory(df)

    return dr


# count() {{{1
def count(df, columns=None, totals_name='n', sort_values=False, percent=True,
          cum_percent=True, threshold=100, round=2, totals=False,
          reset_index=False):
    ''' For each unique 'value' within selected column(s)
    total count, % count total, cum % count


    Example
    -------
    counts(df, columns=key, sort_values=False, percent=True, cum_percent=True)


    Parameters
    ----------
    df : dataframe reference

    columns: dataframe columns/index to be used in groupby function

    totals_name: str, name of total column, default 'n'

    sort_values: default False, None means use index sort

    percent: provide % total, default True

    cum_percent: provide cum % total, default True

    threshold: filter cum_percent by this value, default 100

    round = round decimals, default 2

    totals: bool, add total column, default False

    reset_index: bool, default False
    '''
    try:
        if isinstance(columns, str):
            p1 = df.groupby(columns).agg(totals=(columns, 'count'))
        elif isinstance(df, pd.Series):
            new_df = df.to_frame()
            columns = new_df.columns.tolist()[0]
            p1 = new_df.groupby(columns).agg(totals=(columns, 'count'))
        elif isinstance(df, pd.DataFrame):
            if columns is not None:
                p1 = df.groupby(columns).agg(totals=(columns[0], 'count'))
            else:
                p1 = df.count().to_frame()

    except (KeyError, AttributeError) as e:
        logger.info(f"Column {columns} not found!")
        return

    if p1.shape[0] > 0:
        p1.columns = [totals_name]

        if sort_values is not None:
            p1.sort_values(by=totals_name, ascending=sort_values, inplace=True)

        if percent:
            func = lambda x : (x*100/x.sum()).round(round)
            p1['%'] = func(p1[totals_name].values)

        if cum_percent:
            p1['cum %'] = (p1[totals_name].cumsum() * 100 / p1[totals_name].sum()).round(round)

            if threshold < 100:
                p1 = p1[p1['cum %'] <= threshold]

        if reset_index:
            p1 = p1.reset_index()

        if totals:
            cols = ['n']
            if percent:
                cols.append('%')

            p1 = adorn(p1, columns=cols, col_row_name='Total',
                       ignore_row_index=reset_index)

    return p1


# sample() {{{1
def sample(df, n=2, frac=None, replace=False, weights=None, random_state=None,
        axis=None, info=True):
    ''' sample: wrapper function rather than using e.g. df.sample()

    Example:
    --------
    sample(df)

    Parameters
    ----------
    df : dataframe

    n : int, optional
        Number of items from axis to return. Cannot be used with `frac`.
        Default = 2 if `frac` = None.
    frac : float, optional
        Fraction of axis items to return. Cannot be used with `n`.
    replace : bool, default False
        Allow or disallow sampling of the same row more than once.
    weights : str or ndarray-like, optional
        Default 'None' results in equal probability weighting.
        If passed a Series, will align with target object on index. Index
        values in weights not found in sampled object will be ignored and
        index values in sampled object not in weights will be assigned
        weights of zero.
        If called on a DataFrame, will accept the name of a column
        when axis = 0.
        Unless weights are a Series, weights must be same length as axis
        being sampled.
        If weights do not sum to 1, they will be normalized to sum to 1.
        Missing values in the weights column will be treated as zero.
        Infinite values not allowed.
    random_state : int or numpy.random.RandomState, optional
        Seed for the random number generator (if int), or numpy RandomState
        object.
    axis : {0 or ‘index’, 1 or ‘columns’, None}, default None
        Axis to sample. Accepts axis number or name. Default is stat axis
        for given data type (0 for Series and DataFrames).

    Returns
    -------
    returns input dataframe using .sample() pandas function
    '''
    if info:
        if isinstance(df, pd.Series):
            logger.info(f'{df.shape[0]} rows')

        if isinstance(df, pd.DataFrame):
            logger.info(f'{df.shape[0]} rows, {df.shape[1]} columns')

    if isinstance(df, pd.Series):
        return df.sample(n=n, frac=frac, replace=replace, weights=weights,
            random_state=random_state, axis=axis).to_frame()

    return df.sample(n=n, frac=frac, replace=replace, weights=weights,
        random_state=random_state, axis=axis)


# columns() {{{1
def columns(df, regex=None, astype='list'):
    ''' for given dataframe, return a dictionary of columns.

    This function is useful when renaming a group of columns

    Example
    -------

    df = pd.DataFrame(None, columns=['item', 'item_desc', 'qty'])
    columns_to_rename = columns(df, 'item')

    : {'item': 'item', 'item_desc': 'item_desc'}

    Parameters
    ----------
    df : dataframe

    regex : regular expression to 'filter' list of returned columns
            default, None

    astype : 'dict' (default), 'list', 'text', 'dataframe'

    Returns
    -------
    column dictionary

    '''
    cols = df.columns.tolist()

    if regex:
        cols = list(filter(re.compile(regex).match, cols))

    if astype == 'dict':
        return {x: x for x in cols}
    elif astype == 'list':
        return cols
    elif astype == 'text':
        return "['" +"', '".join(cols)+ "']"
    elif astype == 'dataframe':
        return pd.DataFrame(cols, columns = ['column_names'])
    elif astype == 'series':
        return pd.Series(cols, name='column_names')


# set_columns() {{{1
def set_columns(df, columns=None):
    '''
    Example:
    --------
    '''
    df.columns = columns

    return df


# clean_columns() {{{1
def clean_columns(df, replace_char=(' ', '_'), title=False, inplace=False):
    '''
    Given dataframe, clean column names, strip blanks, lowercase,
    snake_case and remove awkward characters to allow for easier manipulation.
    (optionally 'title' each column name.)

    Example:
    --------
    df = clean_columns(df)

    Parameters
    ----------
    df : dataframe

    replace_char : a tuple giving the 'from' and 'to' characters
                   to be replaced, default (' ', '_')

    title : True, False - default False
            Title each column name

    inplace : False (default) return new updated dataframe
              True return passed dataframe updated

    Returns
    -------
    pandas DataFrame object
    '''
    df2 = _inplace(df, inplace)

    from_char, to_char = replace_char

    if title:
        f = lambda x: x.strip(''.join(replace_char)).replace(from_char, to_char).replace('.', '').lower().title()
    else:
        f = lambda x: x.strip(''.join(replace_char)).replace(from_char, to_char).replace('.', '').lower()

    df2.columns = list(map(f, df2.columns))

    return df2


# combine_header_rows() {{{1
def combine_header_rows(df, start=0, end=1, delimitter=' ', apply=None,
                        infer_objects=True):
    '''
    Combine header data spread over 1 or more rows into one, returning new header
    and dataframe data passed. Optionally, infers remaining data column data
    types.

    Example:
    --------
    | A     | B      |
    |:------|:-------|
    | Order | Order  |
    | Qty   | Number |
    | 10    | 12345  |
    | 40    | 12346  |

    data = {'A': ['Order', 'Qty', 10, 40],
            'B': ['Order', 'Number', 12345, 12346]}
    df = pd.DataFrame(data)
    df = combine_header_rows(df)
    df

    |   Order Qty |   Order Number |
    |------------:|---------------:|
    |          10 |          12345 |
    |          40 |          12346 |

    Parameters
    ----------
    df: dataframe

    start: int, starting row - default 0

    end: int, ending row to combine, - default 1

    delimitter: str, character to be used to 'join' row values together.
                default is ' '

    apply: str, function to apply to combined heading values - default None
               accepted values are: 'lower', 'title'

    Returns
    -------
    pandas dataframe

    Example
    -------
    %%piper
    df <- pd.read_excel(xl_file, header=None)
    >> combine_header_rows(infer_objects=True)
    >> head(5)

    '''
    data = df.iloc[start].astype(str).values
    logger.debug(data)

    rows = range(start+1, end+1)
    logger.debug(rows)

    for row in rows:
        data = data + delimitter + df.iloc[row].astype(str).values
        logger.debug(data)

    if apply == 'lower':
        df.columns = [x.lower() for x in data]
    elif apply == 'title':
        df.columns = [x.title() for x in data]
    else:
        df.columns = data

    # Read remaining data and reference back to dataframe reference
    df = df.iloc[end + 1:]

    if infer_objects:
        df = df.infer_objects()

    return df


# flatten_cols() {{{1
def flatten_cols(df, join_char='_', remove_prefix=None):
    '''
    For given multi-index dataframe reduce/flatten column names to make it easier to
    refer to the column name references subsequently.

    Example
    -------
    from piper import piper
    from piper.factory import get_sample_sales
    from piper.verbs import *
    %%piper

    get_sample_sales()
    >> group_by(['location', 'product'])
    >> summarise(Total=('actual_sales', 'sum'))
    >> pd.DataFrame.unstack()
    >> flatten_cols(remove_prefix='Total')
    >> pd.DataFrame.reset_index()

    | location   |   Beachwear |   Footwear |   Jeans |   Sportswear |   Tops & Blouses |
    |:-----------|------------:|-----------:|--------:|-------------:|-----------------:|
    | London     |      417048 |     318564 |  432023 |       461916 |           345902 |
    | Milan      |      420636 |     313556 |  471798 |       243028 |           254410 |
    | Paris      |      313006 |     376844 |  355793 |       326413 |           400847 |

    '''
    def flatten(column_string, join_char='_', remove_prefix=None):
        ''' Takes a dataframe column string,
        if it's a tuple (from a groupby/stack/pivot df) returns a
        'joined' string otherwise just return the string.
        '''
        if isinstance(column_string, tuple):
            return_str = join_char.join([str(x) for x in column_string]).strip(join_char)

            if remove_prefix is not None:
                return re.sub(remove_prefix, '', return_str).strip(join_char)

            return return_str

        return column_string

    df.columns = [flatten(x, join_char, remove_prefix) for x in df.columns]

    return df


# trim() {{{1
def trim(df, str_columns=None, inplace=False):
    '''
    strip leading and trailing blanks of all
    string/object based column (row) values

    Parameters
    ----------
    df : pandas dataframe

    str_columns : Optional list of columns to strip.
                  If None, use df.columns.values

    inplace : False (default) return new updated dataframe
              True return passed dataframe updated

    Returns
    -------
    pandas dataframe
    '''
    df2 = _inplace(df, inplace)

    if str_columns is None:
        str_columns = df2.select_dtypes(include='object')

    # duplicate names check
    check_list = {}
    for idx, item in enumerate(df.columns.values):
        if item not in check_list:
            check_list[item] = 1
        else:
            suffix = check_list[item]
            check_list[item] = suffix + 1
            df.columns.values[idx] = df.columns.values[idx] + str(check_list[item])

    for col in str_columns:
        df2[col] = df2[col].astype('str').str.strip()

    return df2


# select() {{{1
def select(df, expr=None, regex=False):
    ''' Based on select() function from R.

    Given dataframe, select column names

    Example:
    --------
    select(df) # select ALL columns

    # select column column listed
    select(df, 'column_name')

    # select columns listed
    select(df, ['column_name', 'other_column'])

    # select ALL columns EXCEPT the column listed (identified by - minus sign prefix)
    select(df, '-column_name')

    # select ALL columns EXCEPT the column specified with a minus sign within the list
    select(df, ['-column_name', '-other_column'])

    # select column range from column up to and including the 'to' column.
    # This is achieved by passing a 'slice' -> e.g. slice('from_col', 'to_col')
    select(df, slice('title', 'isbn'))

    # select using a regex string
    select(df, 'value') -> select fields containing 'value'
    select(df, '^value') -> select fields starting with 'value'
    select(df, 'value$') -> select fields ending with 'value'


    Parameters
    ----------
    expr : str, list - default None
            str - single column (in quotes)
            list -> list of column names (in quotes)

            NOTE: prefixing column name with a minus sign
                  filters out the column from returned list of columns

    regex : boolean, treat column string as a regex
            default False

    Returns
    -------
    pandas DataFrame object
    '''
    returned_column_list = df.columns.tolist()

    try:

        if expr is None:
            return df

        if isinstance(expr, slice):
            return df.loc[:, expr]

        if isinstance(expr, str):

            if regex:
                return df.filter(regex=expr)

            if expr.startswith('-'):
                returned_column_list.remove(expr[1:])
                return df[returned_column_list]
            else:
                if expr not in returned_column_list:
                    raise KeyError(f"Column '{expr}' not found!")

                return df[expr].to_frame()

        if isinstance(expr, list):

            selected_cols = [x for x in expr if not x.startswith('-')]
            if len(selected_cols) > 0:
                return df[selected_cols]

            remove_cols = [x[1:] for x in expr if x.startswith('-')]
            if len(remove_cols) > 0:
                logger.debug(remove_cols)
                logger.debug(len(remove_cols))

                returned_column_list = [x for x in returned_column_list if x not in remove_cols]

                for col in returned_column_list:
                    returned_column_list

                return df[returned_column_list]

    except (ValueError, KeyError) as e:
        logger.info(e)


# rename() {{{1
@wraps(pd.DataFrame.rename)
def rename(df, *args, **kwargs):
    '''

    Example:
    ========
    %%piper

    get_sample_sales()
    >> rename(columns={'product': 'item'})

    '''
    return df.rename(*args, **kwargs)


# rename_axis() {{{1
@wraps(pd.DataFrame.rename_axis)
def rename_axis(df, *args, **kwargs):
    '''
    Example:
    ========
    %%piper

    get_sample_sales()
    >> pivot_table(index=['location', 'product'], values='target_sales')
    >> rename_axis(('AAA', 'BBB'), axis='rows')
    >> head()

    |                          |   target_sales |
    |    AAA          BBB      |                |
    |:-------------------------|---------------:|
    | ('London', 'Beachwear')  |        31379.8 |
    | ('London', 'Footwear')   |        27302.6 |
    | ('London', 'Jeans')      |        28959.8 |
    | ('London', 'Sportswear') |        29466.4 |

    '''
    return df.rename_axis(*args, **kwargs)


# relocate() {{{1
def relocate(df, column=None, loc='last', ref_column=None,
             index=False):
    ''' relocate: move column(s) or index(es) within a dataframe

    Example:
    --------
    %%piper
    get_sample_sales()
    >> pd.DataFrame.set_index(['location', 'product'])
    >> relocate(column='location', loc='after',
                ref_column='product', index=True)
    >> head()


    Parameters
    ----------
    df : dataframe

    column : str or list: column name(s) to be moved

    loc : str, 'first', 'last', 'before', 'after'
          new location, default='last'

    ref_column = None

    index : boolean, default is False (column)
            if True, then the column(s) being moved are
            considered to be row indexes.

    Returns
    -------
    pandas dataframe

    '''
    def sequence(columns, index=False):
        ''' Return column or index sequence '''

        if index:
            return df.reorder_levels(columns)
        else:
            return df[columns]


    if index:
        type_ = 'index(es)'
        df_cols = df.index.names.copy()
    else:
        type_ = 'column(s)'
        df_cols = df.columns.tolist().copy()

    if isinstance(column, str):
        column = [column]

    if isinstance(column, list):
        errors = [x for x in column if x not in df_cols]

        if errors != []:
            logger.info(f'{type_} {errors} not found!')
            return df

    # Remove columns to move from main list
    df_cols = [x for x in df_cols if x not in column]

    if loc == 'first':
        df_cols = column + df_cols
        return sequence(df_cols, index=index)

    elif loc == 'last':
        df_cols = df_cols + column
        return sequence(df_cols, index=index)

    if loc == 'after':
        position = 0
    elif loc == 'before':
        position = len(column)

    new_column_sequence = []
    for col in df_cols:
        if col == ref_column:
            column.insert(position, col)
            new_column_sequence.append(column)
        else:
            new_column_sequence.append(col)

    new_column_sequence = list(flatten(new_column_sequence))

    return sequence(new_column_sequence, index=index)


# drop() {{{1
@wraps(pd.DataFrame.drop)
def drop(df, *args, **kwargs):
    '''
    Example:
    =======
    df <- pd.read_csv('inputs/export.dsv', sep='\t')
    >> clean_columns()
    >> trim()
    >> assign(adast = lambda x: x.adast.astype('category'),
              adcrcd = lambda x: x.adcrcd.astype('category'),
              aduom = lambda x: x.aduom.astype('category'))
    >> drop(columns='adcrcd_1')
    '''
    return df.drop(*args, **kwargs)


# assign() {{{1
def assign(df, *args, str_to_lambdas=True, lambda_var='x', info=False, **kwargs):
    ''' Assign new columns to a DataFrame.

    Returns a new object with all original columns in addition to new ones.
    Existing columns that are re-assigned will be overwritten.

    Parameters
    ----------
    **kwargs : dict of {str: callable or Series}
        The column names are keywords. If the values are
        callable, they are computed on the DataFrame and
        assigned to the new columns. The callable must not
        change input DataFrame (though pandas doesn't check it).
        If the values are not callable, (e.g. a Series, scalar, or array),
        they are simply assigned.

    str_to_lambdas: boolean - default is False.

    lambda_var: str - default is 'x'.

    info: boolean - default is False.
        if True - output contents of kwargs where keyword value is being
        converted from string to lambda function. (For debug purposes)


    Returns
    -------
    DataFrame
        A new DataFrame with the new columns in addition to
        all the existing columns.


    Notes
    -----
    Assigning multiple columns within the same ``assign`` is possible.
    Later items in '\\*\\*kwargs' may refer to newly created or modified
    columns in 'df'; items are computed and assigned into 'df' in order.


    Example:
    --------
    %%piper

    get_sample_sales()
    >> select()
    >> assign(month_plus_one = lambda x: x.month + pd.Timedelta(1, 'D'),
              alternative_text_formula = "x.actual_sales * .2")

    ---
    assign(profit_sales = 'x.actual_profit - x.actual_sales',
           month_value = "x.month.dt.month",
           product_added = "x['product'] + ', TEST'",
           salesthing = "x.target_sales.sum()", info=False)

    ---
    Alternative, use standard lambda convention
    assign(adast = lambda x: x.adast.astype('category'),
           adcrcd = lambda x: x.adcrcd.astype('category'),
           aduom = lambda x: x.aduom.astype('category'))

    '''
    if kwargs:
        for keyword, value in kwargs.items():
            if str_to_lambdas:
                if isinstance(value, str):
                    lambda_str = f"lambda {lambda_var}: " + value
                    function = eval(lambda_str)
                    if info:
                        logger.info(f'keyword {keyword} = {lambda_str}')

                    kwargs[keyword] = function

    return df.assign(*args, **kwargs)


# where() {{{1
@wraps(pd.DataFrame.query)
def where(df, *args, **kwargs):
    '''
    Example:
    ========
    (
    select(customers)
     .pipe(clean_columns)
     .pipe(select, ['client_code', 'establishment_type', 'address_1', 'address_2', 'town'])
     .pipe(where, 'establishment_type != 91')
     .pipe(where, "town != 'LISBOA' & establishment_type != 91")
    )
    '''

    logger.debug(args)
    logger.debug(kwargs)

    args_copy = list(args)
    args_copy[0] = re.sub('\n', '', args_copy[0])

    try:
        return df.query(*args_copy, **kwargs)
    except (pd.core.computation.ops.UndefinedVariableError) as e:
        logger.info("ERROR: External @variable? use: where(..., global_dict=globals())")


# distinct() {{{1
@wraps(pd.DataFrame.drop_duplicates)
def distinct(df, *args, info=True, **kwargs):
    '''
    Example:
    ========
    %%piper
    df >> distinct(['author']) >> head()
    '''
    result = df.drop_duplicates(*args, **kwargs)

    if info:
        if isinstance(df, pd.Series):
            logger.info(f'{df.shape[0]} rows')

        if isinstance(df, pd.DataFrame):
            logger.info(f'{df.shape[0]} rows, {df.shape[1]} columns')

    return result


# stack() {{{1
@wraps(pd.DataFrame.stack)
def stack(df, *args, **kwargs):
    '''
    Example:
    ========
    %%piper
    df >> stack() >> head()
    '''
    return df.stack(*args, **kwargs)


# unstack() {{{1
@wraps(pd.DataFrame.unstack)
def unstack(df, *args, **kwargs):
    '''
    Example:
    ========
    %%piper
    df >> unstack() >> head()
    '''
    return df.unstack(*args, **kwargs)


# set_index() {{{1
@wraps(pd.DataFrame.reset_index)
def set_index(df, *args, **kwargs):
    '''
    Example:
    ========
    %%piper
    df >> set_index() >> head()
    '''
    return df.set_index(*args, **kwargs)


# reset_index() {{{1
@wraps(pd.DataFrame.reset_index)
def reset_index(df, *args, **kwargs):
    '''
    Example:
    ========
    %%piper
    df >> reset_index() >> head()
    '''
    return df.reset_index(*args, **kwargs)


# summarise() {{{1
def summarise(df, *args, **kwargs):
    ''' wrapper for pd.Dataframe.agg() function

    Examples
    --------

    # Syntax 1: column_name = ('existing_column', function)
    # note: below there are some 'common' functions that can
    # be quoted like 'sum', 'mean', 'count', 'nunique' or
    # just state the function name

    %%piper
    get_sample_sales() >>
    group_by('product') >>
    summarise(totval1=('target_sales', sum),
              totval2=('actual_sales', 'sum'))


    # Syntax 2: column_name = pd.NamedAgg('existing_column', function)
    %%piper
    get_sample_sales() >>
    group_by('product') >>
    summarise(totval1=(pd.NamedAgg('target_sales', 'sum')),
              totval2=(pd.NamedAgg('actual_sales', 'sum')))


    # Syntax 3: {'existing_column': function}
                {'existing_column': [function1, function2]}
    %%piper
    get_sample_sales()
    >> group_by('product')
    >> summarise({'target_sales':['sum', 'mean']})

    # Syntax 4: 'existing_column': lambda x: x+1
    # Example below identifies unique products sold by location.
    %%piper
    get_sample_sales() >>
    group_by('location') >>
    summarise({'product': lambda x: set(x.tolist())}) >>
    # explode('product')

    '''
    if not kwargs and len(args) == 0:
        return df.agg('count')

    group_df = df.agg(*args, **kwargs)

    return group_df


# adorn() {{{1
def adorn(df, columns=None, fillna='', col_row_name='All', axis=0, ignore_row_index=False):
    '''
    Based on R function, for given dataframe, add row or column totals.


    Parameters
    ----------
    df - Pandas dataframe

    fillna - fill NaN values (default is '')

    col_row_name - name of row/column title (default 'Total')

    axis - axis to apply total (values: 0 or 'row', 1 or 'column')
           To apply totals to both axes - use 'both'.
           (default is 0)

    ignore_row_index - when concatenating totals, ignore index in both dataframes.
                       (default is False)

    Example
    -------
    import pandas as pd
    from piper.pandas import *

    url = 'https://github.com/datagy/pivot_table_pandas/raw/master/sample_pivot.xlsx'
    df = pd.read_excel(url, parse_dates=['Date'])
    head(df)

    Date        Region          	  Type     Units   	Sales
    2020-07-11  East    Children's Clothing 	18.0      306
    2020-09-23  North   Children's Clothing 	14.0      448

    g1 = df.groupby(['Type', 'Region']).agg(TotalSales=('Sales', 'sum')).unstack()
    g1 = adorn(g1, axis='both').astype(int)
    g1 = flatten_cols(g1, remove_prefix='TotalSales')
    g1

    |                     |   East |   North |   South |   West |    All |
    |:--------------------|-------:|--------:|--------:|-------:|-------:|
    | Children's Clothing |  45849 |   37306 |   18570 |  20182 | 121907 |
    | Men's Clothing      |  51685 |   39975 |   18542 |  19077 | 129279 |
    | Women's Clothing    |  70229 |   61419 |   22203 |  22217 | 176068 |
    | All                 | 167763 |  138700 |   59315 |  61476 | 427254 |

    '''
    # ROW:
    if axis == 0 or axis == 'row' or axis == 'both':
        total_row = ['' for _ in df.index.names]

        # If index is only 1, return that else return tuple of
        # combined multi-index(key)
        index_length = len(df.index.names)
        if index_length == 1:
            total_row = col_row_name
        else:
            total_row[len(df.index.names) - 1] = col_row_name
            total_row = tuple(total_row)

        totals = {}
        if columns is None:
            numeric_columns = df.select_dtypes(include='number').columns
        else:
            if isinstance(columns, str):
                numeric_columns = [columns]
            else:
                numeric_columns = columns

        for column in numeric_columns:
            totals[column] = df[column].sum()

        if ignore_row_index:
            totals = pd.DataFrame(totals, index=range(1))
            df = pd.concat([df, totals], axis=0, ignore_index=ignore_row_index)
            df.iloc[-1, 0] = col_row_name
        else:
            totals = pd.DataFrame(totals, index=[total_row])
            df = pd.concat([df, totals], axis=0, ignore_index=ignore_row_index)

    # COLUMN: Sum numeric column(s) and concatenate result to dataframe
    if axis == 1 or axis == 'column' or axis == 'both':
        totals = pd.DataFrame()

        if columns is None:
            columns = df.select_dtypes(include='number').columns
        else:
            if isinstance(columns, str):
                columns = [columns]

        totals[col_row_name] = df[columns].sum(axis=1)

        df = pd.concat([df, totals], axis=1)

    if fillna is not None:
        df = df.fillna(fillna)

    return df

# order_by() {{{1
@wraps(pd.DataFrame.sort_values)
def order_by(df, *args, **kwargs):
    '''
    %%piper

    get_sample_data()
    >> group_by(['countries', 'regions'])
    >> summarise(totalval1=('values_1', 'sum'))
    >> group_calc(index='countries')
    >> order_by(['countries', '-group%'])
    >> head(8)

    |                      |   totalval1 |   group% |
    |:---------------------|------------:|---------:|
    | ('France', 'West')   |        4861 |    42.55 |
    | ('France', 'North')  |        2275 |    19.91 |
    | ('France', 'East')   |        2170 |    19    |
    | ('France', 'South')  |        2118 |    18.54 |
    | ('Germany', 'North') |        2239 |    30.54 |
    | ('Germany', 'East')  |        1764 |    24.06 |
    | ('Germany', 'South') |        1753 |    23.91 |
    | ('Germany', 'West')  |        1575 |    21.48 |

    '''
    # args is a tuple, therefore need to 'copy' to a list for manipulation
    args_copy = list(args)

    if kwargs.get('by'):
        column_seq = kwargs.get('by')
    else:
        if isinstance(args_copy[0], str) or isinstance(args_copy[0], list):
            column_seq = args_copy[0]

    f = lambda x: True if not x.startswith('-') else False

    if isinstance(column_seq, list):
        sort_seq = [f(x) for x in column_seq]

        # Set sort sequence - only if NOT specified by user
        if kwargs.get('ascending') is None:
            kwargs['ascending'] = sort_seq

        f = lambda ix, x: column_seq[ix] if x else column_seq[ix][1:]
        if args_copy != []:
            args_copy[0] = [f(ix, x) for ix, x in enumerate(sort_seq)]
        else:
            kwargs['by'] = [f(ix, x) for ix, x in enumerate(sort_seq)]

    else:
        # Assume ascending sequence, unless otherwise specified
        if kwargs.get('ascending') is None:

            kwargs['ascending'] = f(column_seq)
            if kwargs['ascending'] == False:
                column_seq = column_seq[1:]

            if args_copy != []:
                args_copy[0] = column_seq
            else:
                kwargs['by'] = column_seq

    return df.sort_values(*args_copy, **kwargs)


# inner_join() {{{1
@wraps(pd.DataFrame.merge)
def inner_join(df, *args, **kwargs):
    '''

    order_data = {'OrderNo': [1001, 1002, 1003, 1004, 1005],
                  'Status': ['A', 'C', 'A', 'A', 'P'],
                  'Type_': ['SO', 'SA', 'SO', 'DA', 'DD']}
    orders = pd.DataFrame(order_data)

    status_data = {'Status': ['A', 'C', 'P'],
                   'description': ['Active', 'Closed', 'Pending']}
    statuses = pd.DataFrame(status_data)

    order_types_data = {'Type_': ['SA', 'SO'],
                        'description': ['Sales Order', 'Standing Order'],
                        'description_2': ['Arbitrary desc', 'another one']}
    types_ = pd.DataFrame(order_types_data)

    %%piper
    orders >> inner_join(types_, suffixes=('_orders', '_types'))

    |   OrderNo | Status   | Type_   | description    | description_2   |
    |----------:|:---------|:--------|:---------------|:----------------|
    |      1001 | A        | SO      | Standing Order | another one     |
    |      1003 | A        | SO      | Standing Order | another one     |
    |      1002 | C        | SA      | Sales Order    | Arbitrary desc  |

    '''
    kwargs['how'] = 'inner'
    logger.debug(f"{kwargs}")

    return df.merge(*args, **kwargs)


# left_join() {{{1
@wraps(pd.DataFrame.merge)
def left_join(df, *args, **kwargs):
    '''

    order_data = {'OrderNo': [1001, 1002, 1003, 1004, 1005],
                  'Status': ['A', 'C', 'A', 'A', 'P'],
                  'Type_': ['SO', 'SA', 'SO', 'DA', 'DD']}
    orders = pd.DataFrame(order_data)

    status_data = {'Status': ['A', 'C', 'P'],
                   'description': ['Active', 'Closed', 'Pending']}
    statuses = pd.DataFrame(status_data)

    order_types_data = {'Type_': ['SA', 'SO'],
                        'description': ['Sales Order', 'Standing Order'],
                        'description_2': ['Arbitrary desc', 'another one']}
    types_ = pd.DataFrame(order_types_data)

    %%piper
    orders >> left_join(types_, suffixes=('_orders', '_types'))

    |   OrderNo | Status   | Type_   | description    | description_2   |
    |----------:|:---------|:--------|:---------------|:----------------|
    |      1001 | A        | SO      | Standing Order | another one     |
    |      1002 | C        | SA      | Sales Order    | Arbitrary desc  |
    |      1003 | A        | SO      | Standing Order | another one     |
    |      1004 | A        | DA      | nan            | nan             |
    |      1005 | P        | DD      | nan            | nan             |

    '''
    kwargs['how'] = 'left'
    logger.debug(f"{kwargs}")

    return df.merge(*args, **kwargs)


# right_join() {{{1
@wraps(pd.DataFrame.merge)
def right_join(df, *args, **kwargs):
    '''

    order_data = {'OrderNo': [1001, 1002, 1003, 1004, 1005],
                  'Status': ['A', 'C', 'A', 'A', 'P'],
                  'Type_': ['SO', 'SA', 'SO', 'DA', 'DD']}
    orders = pd.DataFrame(order_data)

    status_data = {'Status': ['A', 'C', 'P'],
                   'description': ['Active', 'Closed', 'Pending']}
    statuses = pd.DataFrame(status_data)

    order_types_data = {'Type_': ['SA', 'SO'],
                        'description': ['Sales Order', 'Standing Order'],
                        'description_2': ['Arbitrary desc', 'another one']}
    types_ = pd.DataFrame(order_types_data)

    %%piper
    orders >> right_join(types_, suffixes=('_orders', '_types'))

    |   OrderNo | Status   | Type_   | description    | description_2   |
    |----------:|:---------|:--------|:---------------|:----------------|
    |      1002 | C        | SA      | Sales Order    | Arbitrary desc  |
    |      1001 | A        | SO      | Standing Order | another one     |
    |      1003 | A        | SO      | Standing Order | another one     |

    '''
    kwargs['how'] = 'right'
    logger.debug(f"{kwargs}")

    return df.merge(*args, **kwargs)


# outer_join() {{{1
@wraps(pd.DataFrame.merge)
def outer_join(df, *args, **kwargs):
    '''

    order_data = {'OrderNo': [1001, 1002, 1003, 1004, 1005],
                  'Status': ['A', 'C', 'A', 'A', 'P'],
                  'Type_': ['SO', 'SA', 'SO', 'DA', 'DD']}
    orders = pd.DataFrame(order_data)

    status_data = {'Status': ['A', 'C', 'P'],
                   'description': ['Active', 'Closed', 'Pending']}
    statuses = pd.DataFrame(status_data)

    order_types_data = {'Type_': ['SA', 'SO'],
                        'description': ['Sales Order', 'Standing Order'],
                        'description_2': ['Arbitrary desc', 'another one']}
    types_ = pd.DataFrame(order_types_data)

    %%piper
    orders >> outer_join(types_, suffixes=('_orders', '_types'))

    |   OrderNo | Status   | Type_   | description    | description_2   |
    |----------:|:---------|:--------|:---------------|:----------------|
    |      1001 | A        | SO      | Standing Order | another one     |
    |      1003 | A        | SO      | Standing Order | another one     |
    |      1002 | C        | SA      | Sales Order    | Arbitrary desc  |
    |      1004 | A        | DA      | nan            | nan             |
    |      1005 | P        | DD      | nan            | nan             |
    '''
    kwargs['how'] = 'outer'
    logger.debug(f"{kwargs}")

    return df.merge(*args, **kwargs)


# explode() {{{1
@wraps(pd.DataFrame.explode)
def explode(df, *args, **kwargs):

    return df.explode(*args, **kwargs)


# set_grouper() {{{1
def set_grouper(df, index=None, freq='D'):
    ''' For given dataframe and index (str, list),
    return a pd.Grouper object for each defined index.
    '''

    def grouper(df, col, freq):
        ''' return grouper object, and if datetime object set frequency. '''

        if is_period_dtype(df[col].dtype):
            logger.info(f'Warning:: Period data types not yet supported')

        if is_datetime64_any_dtype(df[col].dtype):
           return pd.Grouper(key=col, freq=freq)

        return pd.Grouper(key=col)

    if isinstance(index, str):
        return grouper(df, index, freq)

    if isinstance(index, list):
        return [grouper(df, col, freq) for col in index]


# format_dateindex() {{{1
def format_dateindex(df, freq='D'):
    ''' format dataframe if index key contains a 'frequency'
    parameter to the the requested freq parameter.
    '''
    fmts = {'A': '%Y', 'AS': '%Y', 'Q': '%b %Y', 'M': '%b', 'D': '%Y-%m-%d'}

    # Reset index, cannot work out how to update date index column with
    # index set.
    index = df.index.names

    df = df.reset_index()

    for col in index:
        if is_datetime64_any_dtype(df[col].dtype):
            df[col] = df[col].dt.strftime(fmts.get(freq))

    df = df.set_index(index)

    return df


# group_by() {{{1
# @wraps(pd.DataFrame.groupby)
def group_by(df, *args, freq='D', **kwargs):
    ''' wrapper for pd.Dataframe.groupby() function

    Example:
    ========
    %%piper

    get_sample_data()
    >> where("ids == 'A'")
    >> where("values_1 > 300 & countries.isin(['Italy', 'Spain'])")
    >> group_by(['countries', 'regions'])
    >> summarise(total_values_1=pd.NamedAgg('values_1', 'sum'))

    '''
    args_copy = list(args)
    logger.debug(args_copy)
    logger.debug(kwargs)

    if kwargs.get('by') != None:
        index = set_grouper(df, kwargs.get('by'), freq=freq)
        kwargs['by'] = index

    if len(args) > 0:
        args_copy[0] = set_grouper(df, args[0], freq=freq)

    df = df.groupby(*args_copy, **kwargs)

    return df


# pivot_table() {{{1
@wraps(pd.DataFrame.pivot_table)
def pivot_table(df, *args, freq='M', format_date=False, **kwargs):
    '''
    Example:
    ========

    '''
    logger.debug(args)
    logger.debug(kwargs)

    if kwargs.get('index') != None:
        index = set_grouper(df, kwargs.get('index'), freq=freq)
        kwargs['index'] = index

    df = pd.pivot_table(df, *args, **kwargs)

    if format_date:
        df = format_dateindex(df, freq=freq)

    return df


# transform() {{{1
def transform(df, index=None, **kwargs):
    ''' Add a 'grouped' calculation.
    Transform is based on the pandas .transform() function.
    Based on the given dataframe and grouping index, creates new aggregated column
    values using the list of keyword/value arguments (kwargs) supplied.

    Example #1 - Calculate group percentage value

    %%piper
    get_sample_data() >>
    group_by(['countries', 'regions']) >>
    summarise(TotSales1=('values_1', 'sum'))

    |                      |   TotSales1 |
    |:---------------------|------------:|
    | ('France', 'East')   |        2170 |
    | ('France', 'North')  |        2275 |
    | ('France', 'South')  |        2118 |
    | ('France', 'West')   |        4861 |
    | ('Germany', 'East')  |        1764 |
    | ('Germany', 'North') |        2239 |
    | ('Germany', 'South') |        1753 |
    | ('Germany', 'West')  |        1575 |

    To add a group percentage based on the countries:

    %%piper
    get_sample_data() >>
    group_by(['countries', 'regions']) >>
    summarise(TotSales1=('values_1', 'sum')) >>
    transform(index='countries', g_percent=('TotSales1', 'percent')) >>
    head(8)

    |                      |   TotSales1 |   g_percent |
    |:---------------------|------------:|------------:|
    | ('France', 'East')   |        2170 |       19    |
    | ('France', 'North')  |        2275 |       19.91 |
    | ('France', 'South')  |        2118 |       18.54 |
    | ('France', 'West')   |        4861 |       42.55 |
    | ('Germany', 'East')  |        1764 |       24.06 |
    | ('Germany', 'North') |        2239 |       30.54 |
    | ('Germany', 'South') |        1753 |       23.91 |
    | ('Germany', 'West')  |        1575 |       21.48 |


    Parameters
    ----------
    df - dataframe to calculate grouped value

    index - grouped column(s) (str or list) to be applied as grouping index.

    kwargs - similar to 'assign', keyword arguments to be
            assigned as dataframe columns containing, tuples
            of column_name and function
            e.g. new_column=('existing_col', 'sum')

            if no kwargs supplied - calculates the group percentage ('g%')
            using the first index column as index key and the first column
            value(s).

    Returns
    -------
    original dataframe with additional grouped calculation column

    '''
    if index is None:
        index = df.index.names[0]

    def check_builtin(function):
        ''' for given function, check if its one of the
        built-in ones
        '''
        built_in_func = {'percent': lambda x: (x * 100 / x.sum()).round(2),
                         'rank': lambda x: x.rank(method='dense', ascending=True),
                         'rank_desc': lambda x: x.rank(method='dense', ascending=False)}

        built_in = built_in_func.get(function, None)
        if built_in:
            return built_in

        return function

    if kwargs != {}:
        for keyword, value in kwargs.items():
            if isinstance(value, tuple):
                column, function = value
                f = check_builtin(function)
                df[keyword] = df.groupby(index)[column].transform(f)
    else:
        # If no kwargs passed, default to group % on
        # first column in dataframe
        column, value = 'g%', df.columns[0]
        func = check_builtin('percent')
        df[column] = df.groupby(index)[value].transform(func)

    return df


# add_formula() {{{1
def add_formula(df, column_name='xl_calc', formula='=CONCATENATE(A{row}, B{row}, C{row})',
                  offset=2, loc='last', inplace=False):

    ''' add Excel (xl) formula column

    Examples
    --------
    formula = '=CONCATENATE(A{row}, B{row}, C{row})'
    add_formula(df, column_name='X7', loc=2, formula=formula, inplace=True)


    Parameters
    ----------
    df : pandas dataframe

    column_name : the column name to be associated with the column formula
                  values, default 'xl_calc'

    formula : Excel formula to be applied

              e.g. '=CONCATENATE(A{row}, B{row}, C{row})'
              {row} is the defined replacement variable which will be replaced
              with actual individual row value.

    offset : starting row value, default=2

    loc : column position to insert excel formula - default 'last'.

    inplace : If True, do operation inplace.

    Returns
    -------
    pandas dataframe
    '''
    df2 = _inplace(df, inplace)

    col_values = []
    for x in range(offset, df.shape[0] + offset):
        repl_str = re.sub('{ROW}', str(x), string=formula, flags=re.I)
        col_values.append(repl_str)

    if loc == 'last':
        loc = df2.shape[1]

    df2.insert(loc=loc, column=column_name, value=col_values)

    return df2


# duplicated() {{{1
def duplicated(df=None, subset=None, keep=False, sort=True,
               column='duplicate', loc='first', ref_column=None,
               unique_only=False):
    ''' get dataframe with identified duplicate data


    Parameters
    ----------
    df : pandas dataframe

    subset : column label or sequence of labels, required
             Only consider certain columns for identifying duplicates
             Default None - consider ALL dataframe columns

    keep : {‘first’, ‘last’, False}, default ‘first’
            first : Mark duplicates as True except for the first occurrence.
            last : Mark duplicates as True except for the last occurrence.
            False : Mark all duplicates as True.

    sort : True, False
           if True sort returned dataframe using subset fields as key

    column : insert a column name identifying whether duplicate (True/False)
            , default 'duplicate'

    loc : str - Position to insert duplicated column indicator.
                'first', 'last', 'before', 'after',
                default='first'

    ref_column = None


    Returns
    -------
    pandas dataframe

    '''
    dx = df.copy(deep=True)

    if subset is None:
        subset = dx.columns.tolist()

    duplicate_rows = dx.duplicated(subset, keep=keep)

    if column is not None:
        dx[column] = duplicate_rows

        if loc is not None:
           dx = relocate(dx, column=column, loc=loc, ref_column=ref_column)

    if unique_only:
        dx = dx[~duplicate_rows]

    if sort:
        dx = dx.sort_values(subset)

    return dx


# overlaps() {{{1
def overlaps(df, effective='effective_date', expired='expiry_date', price='price',
                     unique_cols=['customer#', 'item'],
                     overlap_name='overlaps', price_diff_name='price_diff',
                     loc='last'):
    ''' Analyse dataframe rows with overlapping dates and optionally different prices

    Example:
    --------
    customer_prices_ovl = overlaps(customer_prices, loc=4)

    Parameters
    ----------
    df :
        dataframe

    effective:
        column that defines start/effective date, default 'effective_date'

    expired:
        column that defines end/expired date, default 'expiry_date'

    price: (bool, str)
        column that defines price, default - False (no check performed)
        If provided, price difference check will be performed.

    unique_cols:
        columns that uniquely identify rows

    overlap_name:
        name of overlapping column, default 'overlaps'

    price_diff_name:
        name of price difference column, default 'price_diff'

    loc : int, 'first', 'last'
        new location (zero based), default='last'

    Returns
    -------
    pandas dataframe
    '''
    if loc == 'first':
        loc = 0

    if loc == 'last':
        loc = df.shape[1]+1

    check_cols = [effective, expired, price] + unique_cols
    if not price:
        check_cols = [effective, expired] + unique_cols

    missing_cols = [col for col in check_cols if col not in df.columns.tolist()]
    if len(missing_cols) > 0:
        raise KeyError(f"Column(s) '{missing_cols}' not found in dataframe")

    # Drop given df's original index, add fresh index as comparison column.
    dfa = df.reset_index(drop=True).reset_index(drop=False)
    dfb = dfa.copy(deep=True)

    df_merge = (dfa.merge(dfb, how='left', on=unique_cols))

    criteria_1 = df_merge[f'{expired}_x'] >= df_merge[f'{effective}_y']
    criteria_2 = df_merge[f'{effective}_x'] <= df_merge[f'{expired}_y']
    criteria_3 = df_merge.index_x != df_merge.index_y

    overlap_data = ((criteria_1) & (criteria_2) & (criteria_3))
    df_merge.insert(loc=loc, column=overlap_name, value=overlap_data)

    if price:
        criteria_4 = df_merge[f'{price}_x'] != df_merge[f'{price}_y']
        price_diff = ((criteria_1) & (criteria_2) & (criteria_3) & (criteria_4))
        df_merge.insert(loc=loc, column=price_diff_name, value=price_diff)

    # sort by given unique key
    df_merge = df_merge[df_merge[overlap_name] == True].sort_values(unique_cols)

    # drop merged fields '_y'
    cols = [x for x in df_merge if '_y' in x]
    df_merge.drop(columns=cols, inplace=True)

    # drop index column
    df_merge.drop(columns='index_x', inplace=True)

    # remove _x suffix from remaining fields
    cols = {x: x.replace('_x', '') for x in df_merge if '_x' in x}
    df_merge.rename(columns=cols, inplace=True)

    return df_merge.reset_index(drop=True)


# memory() {{{1
def memory(df):
    '''
    For given data frame, calculate actual consumed memory in Mb
    '''
    memory = df.memory_usage(deep=True).sum()
    memory = round(memory / (1024 * 1024), 2)

    msg = f'Dataframe consumes {memory} Mb'
    logger.info(msg)

    return df


# has_special_chars() {{{1
def has_special_chars(df, col_name):
    ''' For given dataframe, identify whether column
    name given contains special (non-alphanumeric) characters.

    Parameters
    ----------
    df : pandas dataframe
    col_name : name of column to be checked

    Returns
    -------
    pandas dataframe
    '''
    pattern = r'[a-zA-Z0-9]*$'
    df['item_special_chars'] = ~df[col_name].str.match(pattern)

    return df


# to_csv() {{{1
@wraps(pd.DataFrame.to_csv)
def to_csv(df, *args, **kwargs):

    return df.to_csv(*args, **kwargs, index=False)


# to_tsv() {{{1
@wraps(pd.DataFrame.to_csv)
def to_tsv(df, file_name, sep='\t'):
    ''' pd.write_csv wrapper function '''

    file_name = _file_with_ext(file_name, extension='.tsv')

    rows = df.shape[0]
    if rows > 0:
        df.to_csv(file_name, sep=sep, index=False)
        logger.info(f'{file_name} exported, {df.shape[0]} rows.')


# to_parquet() {{{1
@wraps(pd.DataFrame.to_parquet)
def to_parquet(df, *args, **kwargs):

    return df.to_parquet(*args, **kwargs)

# to_excel() {{{1
@wraps(WorkBook)
def to_excel(df, file_name=None, *args, **kwargs):

    WorkBook(file_name=file_name, sheets=df, *args, **kwargs)


def _inplace(df, inplace=False):
    ''' helper function to return either a copy of
    dataframe or a reference to the passed dataframe
    '''

    if inplace:
        return df
    else:
        return df.copy(deep=True)
