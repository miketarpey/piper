import re
import pandas as pd
import numpy as np
import logging
from functools import wraps
from piper.io import _file_with_ext
from piper.decorators import shape
from pandas.core.common import flatten
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
def info(df, include_dupes=False, filter_na_cols=False, memory_info=True):
    ''' Alternative to df.info(), show all columns that have invalid data
_
    Parameters
    ----------
    df : dataframe

    include_dupes:  boolean - include column showing the numbers of duplicate records
                    default: False

    filter_na_cols: boolean - filter out only columns which contain nulls/np.nan
                    default: False

    memory: boolean - If True, show memory returned dataframe consumption

    '''
    null_col_list = [df[col].isnull().sum() for col in df]
    na_col_list = [df[col].isna().sum() for col in df]
    dtypes_col_list = [df[col].dtype for col in df]
    nunique_list = [df[col].nunique() for col in df]
    dupes_col_list = [df.duplicated(col, keep=False).sum() for col in df]
    total_list = [df[col].value_counts(dropna=False).sum() for col in df]

    if include_dupes:
        dicta = {'columns': df.columns.values, 'data_type': dtypes_col_list,
                 'isna': na_col_list, 'isnull': null_col_list, 'unique': nunique_list,
                 'dupes': dupes_col_list, 'total_count': total_list}
    else:
        dicta = {'columns': df.columns.values, 'data_type': dtypes_col_list,
                 'unique': nunique_list, 'isna': na_col_list, 'isnull': null_col_list,
                 'total_count': total_list}

    dr = pd.DataFrame.from_dict(dicta)

    if filter_na_cols:
        return dr.query('isna > 0')

    if memory_info:
        memory(df)

    return dr


# count() {{{1
def count(df, column=None, normalize=False, sort_values=False,  bins=None,
          dropna=False, add_total=False, totals='n', percent=False,
          cum_percent=False, threshold=100, round=2):
    ''' fancy version of pd.Series.value_counts() with totals, %, cum %

    Example
    -------
    count(df, 'price', sort_values=False, add_total=True
          dropna=True, bins=5, ascending=True, percent=False, cum_percent=True)

    count(customer_prices_merge, 'adjustment', normalize=True,
      sort_values=False, add_total=True, dropna=True,
      ascending=True, percent=False, cum_percent=False).round(2)

       	adjustment 	total rows
    0 	PTCUSPRC 	0.90
    1 	PTCUSLOS 	0.06
    2 	PTCUSPND 	0.04
    3 	Total   	1.00

    Parameters
    ----------

    df : dataframe, or series to be referenced

    column : str, column name to analyse, default None

    normalize: bool, default False
        If True then the object returned will contain the relative frequencies of the unique values.

    sort_values: Sort by frequencies, default False, None means use index sort

    bins: int, optional
        Rather than count values, group them into half-open bins, a convenience for pd.cut,
        only works with numeric data.

    dropna: bool, default True
        Don’t include counts of NaN.

    add_total: bool, add total column, default False

    totals: str, name of total column, default 'n'

    percent: provide % total, default False

    cum_percent: provide cum % total, default False

    threshold: filter cum_percent by this value, default 100

    round: round decimals, default 2

    Returns
    -------
    pandas dataframe
    '''
    if isinstance(df, pd.DataFrame):
        if column is None:
            df = df.count().to_frame()
            df.columns = [totals]
            df = df.reset_index(drop=False)
            return df

    if isinstance(df, pd.Series):
        df = df.to_frame()
        column = df.columns.tolist()[0]

    if sort_values:
        df = (df[column].value_counts(normalize=normalize, ascending=sort_values,
                                      bins=bins, dropna=dropna)
                        .to_frame())
    else:
        df = (df[column].value_counts(normalize=normalize, bins=bins, dropna=dropna)
                        .to_frame())

    df.columns = [totals]

    if percent:
        df['%'] = (df[totals] * 100 / df[totals].sum()).round(round)

    if cum_percent:
        df['cum %'] = (df[totals].cumsum() * 100 / df[totals].sum()).round(round)

        if threshold < 100:
            df = df[df['cum %'] <= threshold]

    df.index.names = [column]

    if add_total:
        df = adorn(df, col_row_name='Total')
        if percent:
            df.loc['Total', '%'] = df['%'].sum()

    if not normalize:
        df[[totals]] = df[[totals]].astype(int)

    df = df.reset_index(drop=False)

    return df


# counts() {{{1
def counts(df, columns=None, totals='n', sort_values=False,
          percent=False, cum_percent=False, threshold=100, round=2,
          reset_index=False):
    ''' Similar to count() function but with multiple keys.
    Provides total count, % count total, cum % count total.

    Example
    -------
    counts(df, columns=key, sort_values=False, percent=True, cum_percent=True)

    Parameters
    ----------

    df : dataframe reference

    columns : dataframe columns/index to be used in groupby function

    totals: str, name of total column, default 'n'

    sort_values: default False, None means use index sort

    percent: provide % total, default False

    cum_percent: provide cum % total, default False

    threshold: filter cum_percent by this value, default 100

    round = round decimals, default 2

    reset_index: bool, default False
    '''
    if isinstance(df, pd.DataFrame):
        if columns is None:
            df = df.count().to_frame()
            df.columns = [totals]
            df = df.reset_index(drop=False)
            return df

    try:
        if isinstance(columns, str):
            p1 = df.groupby(columns).agg(totals=pd.NamedAgg(columns, 'count'))
        else:
            p1 = df.groupby(columns).agg(totals=pd.NamedAgg(columns[0], 'count'))
    except KeyError as e:
        logger.info(f'Invalid key {e}')
        return

    if p1.shape[0] > 0:
        p1.columns = [totals]

        if sort_values is not None:
            p1.sort_values(by=totals, ascending=sort_values, inplace=True)

        if percent:
            func = lambda x : (x*100/x.sum()).round(round)
            p1['%'] = func(p1[totals].values)

        if cum_percent:
            p1['cum %'] = (p1[totals].cumsum() * 100 / p1[totals].sum()).round(round)

            if threshold < 100:
                p1 = p1[p1['cum %'] <= threshold]

    if reset_index:
        p1 = p1.reset_index()

    return p1


# sample() {{{1
def sample(df, n=2, frac=None, replace=False, weights=None, random_state=None,
        axis=None, info=True):
    ''' head: R like function, easier to use and control (than say df.sample())

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
    if inplace:
        df2 = df
    else:
        df2 = df.copy(deep=True)

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
    url = 'https://github.com/datagy/pivot_table_pandas/raw/master/sample_pivot.xlsx'
    df = pd.read_excel(url, parse_dates=['Date'])
    head(df)

    Date       Region           	  Type Units   	Sales
    2020-07-1  East    Children's Clothing 	18.0      306
    2020-09-2  North   Children's Clothing 	14.0      448

    g1 = df.groupby(['Type', 'Region']).agg(TotalUnits=('Units', 'count')).unstack()
    g1 = adorn(g1, axis='both').astype(int)
    head(g1, g1.shape[0])

    TotalUnits 	                                          Total
    Region 	                East 	North 	South 	West
    Type
    Children's Clothing 	113 	85  	45  	42  	285
    Men's Clothing      	122 	0   	39  	41  	202
    Women's Clothing    	176 	142 	53  	53  	424
    Total               	411 	227 	137 	136 	911

    g1 = flatten_cols(g1)
    g1.columns = [x.replace('TotalUnits_', '') for x in g1.columns]
    head(g1, g1.shape[0])

                             TotalUnits_East 	 TotalUnits_North 	 TotalUnits_South 	 TotalUnits_West 	 Total
    Type
    Children's Clothing 	 113             	 85               	 45               	 42              	 285
    Men's Clothing      	 122             	 0                	 39               	 41              	 202
    Women's Clothing    	 176             	 142              	 53               	 53              	 424
    Total               	 411             	 227              	 137              	 136             	 911
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
    if inplace:
        df2 = df
    else:
        df2 = df.copy(deep=True)

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
def select(df, expr=None):
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

    # select ALL columns EXCEPT the column listed
    select(df, ['-column_name', '-other_column'])

    # select from one column upto and including to column
    # by passing a slice
    select(df, slice('title', 'isbn'))

    # Note: You can also pass str functions with df.columns
    select(df, df.columns.str.startswith('product'))
    select(df, df.columns.str.endswith('product'))
    select(df, df.columns.str.contains('product'))

    Parameters
    ----------
    expr : default None -> select all fields
           list -> used passed list of column names

           NOTE: prefixing column name with a minus sign
                 filters out the column from returned list of columns

    Returns
    -------
    pandas DataFrame object
    '''
    returned_column_list = df.columns.tolist()

    try:

        if expr is None:
            return df

        if isinstance(expr, np.ndarray):
            return df.loc[:, expr]

        if isinstance(expr, slice):
            return df.loc[:, expr]

        if isinstance(expr, str):

            if expr.startswith('-'):
                returned_column_list.remove(expr[1:])
                return df[returned_column_list]
            else:
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
        return None

# contains() {{{1
@wraps(pd.Series.str.contains)
def contains(df, *args, **kwargs):
    '''
    Example:
    ========
    df = get_items()
    (select(df)
     .pipe(contains, '(code|technical)')
     .pipe(clean_columns)
     .pipe(group_by, ['product_code'])
     .pipe(summarise, total=pd.NamedAgg('product_code', 'count'))
     .pipe(order_by, 'product_code', ascending=False)
     .pipe(head)
    )
    '''
    return df[df.columns[df.columns.str.contains(*args, **kwargs)]]

# rename() {{{1
@wraps(pd.DataFrame.rename)
def rename(df, *args, **kwargs):
    '''
    Example:
    ========
    (select(df)
     .pipe(counts, 'group_code')
     .pipe(where, "group_code.isin(['P01', 'P15'])")
     .pipe(rename, columns={'count': 'total_rows'})
     .reset_index()
    )
    '''
    return df.rename(*args, **kwargs)


# rename_axis() {{{1
@wraps(pd.DataFrame.rename_axis)
def rename_axis(df, *args, **kwargs):
    '''
    Example:
    ========
    %%piper
    abc <- resample_pivot(df, index=index, grouper=[grouper, grouper2], rule=rule) >>
    rename_axis(index_names, axis='rows') >>

    assign(reg_totval2_percent = piper_group_percent_example ) >>
    head(6)
    '''
    return df.rename_axis(*args, **kwargs)


# relocate() {{{1
def relocate(df, column=None, loc='last', ref_column=None):
    ''' relocate: move column within a dataframe

    Example:
    --------
    %%piper
    df
    >> relocate('FirstName', 'after', 'Department')
    >> head()

    Parameters
    ----------
    df : dataframe

    column : str or list: column name(s) to be moved

    loc : str, 'first', 'last', 'before', 'after'
          new location, default='last'

    ref_column = None

    Returns
    -------
    pandas dataframe

    '''
    df_cols = df.columns.tolist().copy()

    if isinstance(column, str):
        column = [column]

    if isinstance(column, list):
        errors = [x for x in column if x not in df_cols]
        if errors != []:
            logger.info(f'column(s) {errors} not found!')
            return df

    # Remove columns to move from main list
    df_cols = [x for x in df_cols if x not in column]

    if loc == 'first':
        df_cols = column + df_cols
        return df[df_cols]

    elif loc == 'last':
        df_cols = df_cols + column
        return df[df_cols]

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

    return df[new_column_sequence]


# move_index() {{{1
def move_index(df, column=None, loc='last', ref_column=None):
    ''' move_index: move column within a mult-index dataframe

    Example:
    --------
    %%piper

    gx <- df
    >> where("Date >= '2010-01-01'")
    >> group_by(['Department', 'Date', 'Job'])
    >> summarise(AvgSalary=('Salary', 'mean'))
    >> order_by('AvgSalary', ascending=False).round(0).astype(int)
    >> move_index('Department', 'after', 'Job')
    >> adorn(ignore_row_index=False)

    Parameters
    ----------
    df : dataframe

    column : str or list: index name(s) to be moved

    loc : int, 'first', 'last', 'before', 'after'
          new location, default='last'

    ref_column = None

    Returns
    -------
    pandas dataframe

    '''
    df_cols = list(df.index.names).copy()

    if isinstance(column, str):
        column = [column]

    if isinstance(column, list):
        errors = [x for x in column if x not in df_cols]
        if errors != []:
            logger.info(f'indices {errors} not found!')
            return df

    # Remove columns to move from main list
    df_cols = [x for x in df_cols if x not in column]

    if loc == 'first':
        df_cols = column + df_cols
        return df[df_cols]

    elif loc == 'last':
        df_cols = df_cols + column
        return df[df_cols]

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

    return df.reorder_levels(new_column_sequence)


# insert() {{{1
@wraps(pd.DataFrame.insert)
def insert(df, *args, **kwargs):
    '''
    '''
    df.insert(*args, **kwargs)

    return df

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
@wraps(pd.DataFrame.assign)
def assign(df, *args, **kwargs):
    '''
    Example:
    ========
    (get_sample_data()
     .pipe(where, "ids == 'A'")
     .pipe(where, "values_1 > 300 & countries.isin(['Italy', 'Spain'])")
     .pipe(assign, new_field=lambda x: x.countries.str[:3]+x.regions,
                   another=lambda x:3*x.values_1)
    )
    '''
    logger.debug(kwargs)
    logger.debug(args)
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
def distinct(df, *args, **kwargs):
    '''
    Example:
    ========
    %%piper
    df >> distinct(['author']) >> head()
    '''
    return df.drop_duplicates(*args, **kwargs)


# group_by() {{{1
# @wraps(pd.DataFrame.groupby)
def group_by(df, *args, **kwargs):
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
    return df.groupby(*args, **kwargs)


# summarise() {{{1
# @wraps(pd.DataFrame.agg)
def summarise(df=None, *args, **kwargs):
    ''' wrapper for pd.Dataframe.agg() function

    Examples
    ========

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
    if kwargs == {} and args == ():
        return df.agg('count')

    group_df = df.agg(*args, **kwargs)

    return group_df


# adorn() {{{1
def adorn(df, fillna=None, col_row_name='All', columns=None, axis=0, ignore_row_index=False):
    '''
    Based on R function, for given dataframe, add row or column totals.

    Parameters
    ----------

    df - Pandas dataframe

    fillna - fill NaN values (default None)

    col_row_name - name of row/column title (default 'Total')

    axis - axis to apply total (values: 0 or 'row', 1 or 'column')
           To apply totals to both axes - use 'both'.
           (default is 0)

    ignore_row_index - when concatenating totals, ignore index in both dataframes.
                       (default is False)

    Example
    -------

    url = 'https://github.com/datagy/pivot_table_pandas/raw/master/sample_pivot.xlsx'
    df = pd.read_excel(url, parse_dates=['Date'])
    head(df)

    Date        Region          	  Type     Units   	Sales
    2020-07-11  East    Children's Clothing 	18.0      306
    2020-09-23  North   Children's Clothing 	14.0      448

    g1 = df.groupby(['Type', 'Region']).agg(TotalSales=('Sales', 'sum')).unstack()
    g1 = adorn(g1, axis='both').astype(int)#.reset_index()
    g1

 	TotalSales                                          	Total
    Region              	East 	North 	South 	West
    Type
    Children's Clothing 	45849 	37306 	18570 	20182 	121907
    Men's Clothing      	51685 	39975 	18542 	19077 	129279
    Women's Clothing    	70229 	61419 	22203 	22217 	176068
    Total               	167763 	138700 	59315 	61476 	427254
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
    return df.sort_values(*args, **kwargs)


# inner_join() {{{1
@wraps(pd.DataFrame.merge)
def inner_join(df, *args, **kwargs):
    '''
    '''
    kwargs['how'] = 'inner'
    logger.debug(f"{kwargs}")

    return df.merge(*args, **kwargs)


# left_join() {{{1
@wraps(pd.DataFrame.merge)
def left_join(df, *args, **kwargs):
    '''
    '''
    kwargs['how'] = 'left'
    logger.debug(f"{kwargs}")

    return df.merge(*args, **kwargs)


# right_join() {{{1
@wraps(pd.DataFrame.merge)
def right_join(df, *args, **kwargs):
    '''
    '''
    kwargs['how'] = 'right'
    logger.debug(f"{kwargs}")

    return df.merge(*args, **kwargs)


# outer_join() {{{1
@wraps(pd.DataFrame.merge)
def outer_join(df, *args, **kwargs):
    '''
    '''
    kwargs['how'] = 'outer'
    logger.debug(f"{kwargs}")

    return df.merge(*args, **kwargs)


# lookup() {{{1
def lookup(df, df2, columns=None, loc='last', ref_column=None, *args, **kwargs):
    '''
    Given primary and secondary dataframes, merge and return data.
    NOTE:: If type of join ('how') not passed, defaulted to 'left' join.

    Parameters
    ----------
    df : primary pandas dataframe

    df2 : secondary dataframe to be joined/merged with

    columns : (str, list) optionally move column(s) in the returned merged data frame
                    'before', 'after' a certain 'reference column' to be returned with
                    primary dataframe.

                    Default is None (return columns in the default sequence
                    provided by the merge function.

    loc : str, 'first', 'last', 'before', 'after'
          new location, default='last'

    ref_column = (str) reference column to place moved columns before/after
                 , default is None

    *args, kwargs : to be passed to pd.DataFrame.merge()

    Returns
    -------
    Pandas dataframe

    Example
    -------
    order_data = {
        'OrderNo': [1001, 1002, 1003, 1004, 1005],
        'Status': ['A', 'C', 'A', 'A', 'P'],
        'Type_': ['SO', 'SA', 'SO', 'DA', 'DD']
    }

    orders = pd.DataFrame(order_data)

    status_data = {
        'Status': ['A', 'C', 'P'],
        'description': ['Active', 'Closed', 'Pending']
    }

    statuses = pd.DataFrame(status_data)

    order_types_data = {
        'Type_': ['SA', 'SO'],
        'description': ['Sales Order', 'Standing Order'],
        'description_2': ['Arbitrary desc', 'another one']
    }

    types_ = pd.DataFrame(order_types_data)

    %%piper
    orders
    >> lookup(types_)
    >> lookup(statuses, on='Status', columns='description_y', loc='after', ref_column='Status')

    OrderNo 	Status 	description_y 	Type_ 	description_x 	description_2
    1001 	A 	Active 	        SO 	Standing Order 	another one
    1002 	C 	Closed 	        SA 	Sales Order 	Arbitrary desc
    1003 	A 	Active 	        SO 	Standing Order 	another one
    1004 	A 	Active 	        DA 	NaN 	        NaN
    1005 	P 	Pending 	DD 	NaN 	        NaN
    '''
    if kwargs.get('how') is None:
        kwargs['how'] = 'left'
        logger.info(f"merge function 'how' parameter defaulted to '{kwargs.get('how')}'")

    merged_df = df.merge(df2, *args, **kwargs)

    # If columns specified, move them to requested dataframe column position.
    if columns is not None:
        merged_df = relocate(merged_df, column=columns, loc=loc, ref_column=ref_column)

    return merged_df


# explode() {{{1
@wraps(pd.DataFrame.explode)
def explode(df, *args, **kwargs):

    return df.explode(*args, **kwargs)


# pivot_table() {{{1
@wraps(pd.DataFrame.pivot_table)
def pivot_table(df, *args, **kwargs):
    '''
    Example:
    ========
    '''
    return df.pivot_table(*args, **kwargs)


# resample_groupby() {{{1
def resample_groupby(df, index=None, grouper=None, rule='M'):
    ''' resample dataframe with date index using groupby method.

    Parameters
    ----------

    df : dataframe reference

    index : dataframe index to be used in groupby function

    grouper : pd.grouper (date) definition object
              Note: Can accept list of grouper objects too
              e.g. [grouper, grouper2]

    rule : date frequency string value
           e.g. MonthEnd 	'M' 	calendar month end
                MonthBegin 	'MS' 	calendar month begin
                QuarterEnd 	'Q' 	calendar quarter end
                QuarterBegin 	'QS' 	calendar quarter begin
                YearEnd 	'A' 	calendar year end
                YearBegin 	'AS' or 'BYS' 	calendar year begin

    see below for more string values:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

    Returns
    -------
    pandas groupby dataframe


    Example #1
    ----------
    # Single grouper index

    rule = 'Q'
    grouper = pd.Grouper(key='dates', freq=rule)

    index = [grouper]
    g1 = resample_groupby(df, index=index, grouper=grouper, rule=rule)


    Example #2
    ----------
    # Multi-index (including grouper)

    rule = 'M'
    grouper = pd.Grouper(key='dates', freq=rule)
    multi_index = ['countries', grouper, 'ids']

    g1 = resample_groupby(df, multi_index, grouper, rule)

    # Tidy axis labels
    g1.index.names = ['Country', 'period', 'ids']
    g1.columns = ['Totals1', 'Totals2']

    '''
    date_format_conversion = {'A': '%Y', 'Q': '%Y %b', 'M': '%Y %b',
                              'D': '%Y-%m-%d'}

    g1 = df.groupby(index).sum()
    g1.reset_index(inplace=True)

    logger.debug(isinstance(grouper, list))

    if isinstance(grouper, list):
        for group_col in grouper:
            g1[group_col.key] = (g1[group_col.key].dt
                                 .strftime(date_format_conversion.get(rule)))
    else:
        g1[grouper.key] = (g1[grouper.key].dt
                           .strftime(date_format_conversion.get(rule)))

    # set index based on multi-index given
    f = lambda x: x.key if isinstance(x, pd.Grouper) else x
    keys = list(map(f, index))
    g1.set_index(keys, inplace=True)

    return g1


# resample_pivot() {{{1
def resample_pivot(df, index=None, grouper=None, rule='M', values=None, aggfunc='sum'):
    ''' resample dataframe with date index using pivot_table method.

    Parameters
    ----------

    df : dataframe reference

    index : dataframe index to be used in groupby function

    grouper : pd.grouper (date) definition object
              Note: Can accept list of grouper objects too
              e.g. [grouper, grouper2]

    rule : date frequency string value
           e.g. MonthEnd 	'M' 	calendar month end
                MonthBegin 	'MS' 	calendar month begin
                QuarterEnd 	'Q' 	calendar quarter end
                QuarterBegin 	'QS' 	calendar quarter begin
                YearEnd 	'A' 	calendar year end
                YearBegin 	'AS' or 'BYS' 	calendar year begin

    see below for more string values:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

    values : see pivot_table values parameter

    aggfunc : see pivot_table aggfunc parameter

    Returns
    -------
    pandas pivot_table dataframe

    Example #1
    ----------
    # resample pivot with grouper object only

    rule = 'M'
    grouper = pd.Grouper(key='dates', freq=rule)
    index=[grouper]
    p1 = resample_pivot(df, index=index, grouper=grouper, rule=rule, values=['values2'])


    Example #2
    ----------
    # resample pivot with multi-index

    rule = 'M'
    grouper = pd.Grouper(key='dates', freq=rule)
    index = [grouper, 'countries', 'ids']
    p1 = resample_pivot(df, index=index, grouper=grouper, rule=rule)

    '''
    date_format_conversion = {'A': '%Y', 'Q': '%Y %b', 'M': '%Y %b',
                              'D': '%Y-%m-%d'}

    p1 = df.pivot_table(index=index, values=values, aggfunc=aggfunc)
    p1.reset_index(inplace=True)

    if isinstance(grouper, list):
        for group_col in grouper:
            p1[group_col.key] = (p1[group_col.key].dt
                                 .strftime(date_format_conversion.get(rule)))
    else:
        p1[grouper.key] = (p1[grouper.key].dt
                           .strftime(date_format_conversion.get(rule)))

    # set index based on multi-index given
    f = lambda x: x.key if isinstance(x, pd.Grouper) else x
    keys = list(map(f, index))
    p1.set_index(keys, inplace=True)

    return p1


# add_group_calc() {{{1
def add_group_calc(df, index=None, column=None, value=None,
                   function='percent', sort_values=None, ascending=None):
    '''
    Add grouped calculation. This function is useful to make group
    calculations within a group of columns.

    Example #1 - Calculate group percentage value

    df = get_sample_data()
    gx = df.groupby(['countries', 'regions']).agg(TotSales1=('values_1', 'sum'))
    gx.head(8)

                        TotSales1
    countries 	regions
    France 	East         2170
    North 	             2275
    South 	             2118
    West 	             4861
    Germany 	East 	     1764
    North 	             2239
    South 	             1753
    West 	             1575

    This can be expresses as follows:

    gx = add_group_calc(gx, index=['countries'],
                        value='TotSales1',
                        function='percent',
                        sort_values=['countries', 'grouped value'],
                        ascending=[True, False])
    gx.head(8)

                         TotSales1   grouped value
    countries 	regions
    France 	West 	      4861           42.55
    North 	              2275           19.91
    East 	              2170           19.00
    South 	              2118           18.54
    Germany 	North 	      2239           30.54
    East 	              1764           24.06
    South 	              1753           23.91
    West 	              1575           21.48

    Example #2 - Calculating total group value

    gx = df.groupby(['countries', 'regions']).agg(TotSales1=('values_1', 'sum'))
    gx = add_group_calc(gx, column='Total_for_group',
                         index=['countries'], value='TotSales1',
                         function=lambda x: x.sum(),
                         sort_values=['countries', 'Total_for_group'],
                         ascending=[True, False])
    gx.head(8)

                         TotSales1 Total_for_group
    countries 	regions
    France 	East 	      2170           11424
    North 	              2275           11424
    South 	              2118           11424
    West 	              4861           11424
    Germany 	East 	      1764            7331
    North 	              2239            7331
    South 	              1753            7331
    West 	              1575            7331

    Parameters
    ----------
    df - dataframe to calculate grouped value

    index - grouped column(s) (str or list) to be applied as grouping index.

    column - column title to be used (default: None -> 'grouped_%')

    function - function to apply in grouping calculation
               (default 'percent')
               built-in functions: 'percent', 'rank'
               (example: np.sum, np.cumsum, np.mean  etc.)

    sort_values - str/list of column names to sort returned dataframe
                  (default None)

    ascending - list of sort sequences to sort returned dataframe
                (default None)

    Returns
    -------
    original dataframe with additional grouped calculation column

    '''
    if index is None:
        index = df.index

    if value is None:
        value = df.columns[0]

    if function == 'percent':

        if column is None:
            column = 'group_%'

        function = lambda x: (x * 100 / x.sum()).round(2)

    if function == 'rank':
        if column is None:
            column = 'rank'

        function = lambda x: x.rank(method='dense', ascending=False)

    df[column] = df.groupby(index)[value].transform(function)

    if sort_values:
        df = df.sort_values(by=sort_values, ascending=ascending)

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
    if inplace:
        df2 = df
    else:
        df2 = df.copy(deep=True)

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

    Example
    -------

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
