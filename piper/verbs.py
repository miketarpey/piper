from datetime import datetime
from pandas.api.types import is_datetime64_any_dtype
from pandas.api.types import is_period_dtype
from pandas.core.common import flatten
import logging
import numpy as np
import pandas as pd
import re

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


# across {{{1
def across(df: pd.DataFrame,
               columns: Union[str, Tuple[str], List[str]] = None,
               function: Callable = None,
               series_obj: bool = False,
               *args, **kwargs) -> pd.DataFrame:
    '''Apply function across multiple columns

    Across allows you to apply a function across a number of columns in one
    statement. Functions can be applied to series values (via apply()) or access
    pd.Series object methods.

    In pandas, to apply the same function (on a Series/columns' values) you
    would normally do something like this:

    .. code-block::

        df['column'].apply(function)
        df['column2'].apply(function)
        df['column3'].apply(function)

    Piper equivalent would be:

    .. code-block::

        across(df, ['column1', 'column2', 'column3'], function)

    You can also work with Series object functions by passing keyword
    series_obj=True. In Pandas, if you wanted to change the dtype of a column
    you would use something like:

    .. code-block::

        df['col'] = df['col'].astype(float)
        df['col2'] = df['col2'].astype(float)
        df['col3'] = df['col3'].astype(float)

    The equivalent with across would be:

    .. code-block::

        df = across(df, ['col', 'col2', 'col3'], function=lambda x: x.astype(float))


    Parameters
    ----------
    df
        pandas dataframe
    columns
        column(s) to apply function.
            - If a list is provided, only the columns listed are affected by the
              function.
            - If a tuple is supplied, the first and second values will
              correspond to the from and to column(s) range used to apply the function to.
    function
        function to be called.
    series_obj
        Default is False.
        True - Function applied at Series or (DataFrame) 'object' level.
        False - Function applied to each Series row values.


    Returns
    -------
    A pandas dataframe


    Examples
    --------
    See below, example apply a function applied to each of the columns row
    values.

    .. code-block:: python

        %%piper

        sample_data()
        >> across(['dates', 'order_dates'], to_julian)
        # Alternative syntax, passing a lambda...
        >> across(['order_dates', 'dates'], function=lambda x: to_julian(x), series_obj=False)
        >> head(tablefmt='plain')

              dates    order_dates  countries    regions    ids      values_1    values_2
         0   120001         120007  Italy        East       A             311          26
         1   120002         120008  Portugal     South      D             150         375
         2   120003         120009  Spain        East       A             396          88
         3   120004         120010  Italy        East       B             319         233

    .. code-block:: python

        %%piper
        sample_data()
        >> across(['dates', 'order_dates'], fiscal_year, year_only=True)
        >> head(tablefmt='plain')

            dates     order_dates    countries    regions    ids      values_1    values_2
         0  FY 19/20  FY 19/20       Italy        East       A             311          26
         1  FY 19/20  FY 19/20       Portugal     South      D             150         375
         2  FY 19/20  FY 19/20       Spain        East       A             396          88
         3  FY 19/20  FY 19/20       Italy        East       B             319         233

    Accessing Series object methods - by passing series_obj=True you can also
    manipulate series object and string vectorized functions (e.g. pd.Series.str.replace())

    .. code-block:: python

        %%piper
        sample_data()
        >> select(['-ids', '-regions'])
        >> across(columns='values_1', function=lambda x: x.astype(int), series_obj=True)
        >> across(columns=['values_1'], function=lambda x: x.astype(int), series_obj=True)
        >> head(tablefmt='plain')

            dates                order_dates          countries      values_1    values_2
         0  2020-01-01 00:00:00  2020-01-07 00:00:00  Italy               311          26
         1  2020-01-02 00:00:00  2020-01-08 00:00:00  Portugal            150         375
         2  2020-01-03 00:00:00  2020-01-09 00:00:00  Spain               396          88
         3  2020-01-04 00:00:00  2020-01-10 00:00:00  Italy               319         233

    '''
    if isinstance(df, pd.Series):
        raise TypeError('Please specify DataFrame object')

    if function is None:
        raise ValueError('Please specify function to apply')

    if isinstance(columns, str):
        if columns not in df.columns:
            raise ValueError(f'column {columns} not found')

    if isinstance(columns, tuple):
        columns = df.loc[:, slice(*columns)].columns.tolist()

    if isinstance(columns, list):
        for col in columns:
            if col not in df.columns:
                raise ValueError(f'column {col} not found')

    if isinstance(columns, str):

        # If not series function (to be applied to series values)
        if not series_obj:
            df[columns] = df[columns].apply(function, *args, **kwargs)
        else:
            df[[columns]] = df[[columns]].apply(function, *args, **kwargs)

    try:
        # No columns -> Apply with context of ALL dataframe columns
        if columns is None:
            df = df.apply(function, *args, **kwargs)

            return df

        # Specified group of columns to update.
        if isinstance(columns, list):

            # Apply function to each columns 'values'
            if not series_obj:
                for col in columns:
                    df[col] = df[col].apply(function, *args, **kwargs)

            # No, user wants to use/access pandas Series object attributes
            # e.g. str, astype etc.
            else:
                df[columns] = df[columns].apply(function, *args, **kwargs)

    except ValueError as e:
        logger.info(e)
        msg = 'Are you trying to apply a function working with Series values(s)? Try series_obj=False'
        raise ValueError(msg)

    except AttributeError as e:
        logger.info(e)
        msg = 'Are you trying to apply a function using Series object(s)? Try series_obj=True'
        raise AttributeError(msg)

    return df


# adorn() {{{1
def adorn(df: pd.DataFrame,
          columns: Union[str, list] = None,
          fillna: Union[str, int] = '',
          col_row_name: str = 'All',
          axis: Union [int, str] = 0,
          ignore_index: bool = False) -> pd.DataFrame:
    '''add totals to a dataframe

    Based on R janitor package function add row and/or column totals to a
    dataframe.

    Examples
    --------

    .. code-block::

        df = sample_matrix(seed=42)
        df = adorn(df, ['a', 'c'], axis='row')
        head(df, 10, tablefmt='plain')

               a       b    c       d        e
        0     15   8.617   16  25.23     7.658
        1      8  25.792   18   5.305   15.426
        2      5   5.343   12  -9.133   -7.249
        3      4  -0.128   13   0.92    -4.123
        4     25   7.742   11  -4.247    4.556
        All   57           70

    .. code-block::

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

                                  East     North     South     West      All
          Children's Clothing    45849     37306     18570    20182   121907
          Men's Clothing         51685     39975     18542    19077   129279
          Women's Clothing       70229     61419     22203    22217   176068
          All                   167763    138700     59315    61476   427254


    Parameters
    ----------
    df
        Pandas dataframe
    columns
        columns to be considered on the totals row. Default None - All columns
        considered.
    fillna
        fill NaN values (default is '')
    col_row_name
        name of row/column title (default 'Total')
    axis
        axis to apply total (values: 0 or 'row', 1 or 'column')
        To apply totals to both axes - use 'both'. (default is 0)
    ignore_index
        default False. When concatenating totals, ignore index in both
        dataframes.


    Returns
    -------
    A pandas DataFrame with additional totals row and/or column total.

    '''
    # ROW:
    if axis == 0 or axis == 'row' or axis == 'both':

        if columns is None:
            numeric_columns = df.select_dtypes(include='number').columns
        else:
            if isinstance(columns, str):
                numeric_columns = [columns]
            else:
                numeric_columns = columns

        totals = {col: df[col].sum() for col in numeric_columns}

        index_length = len(df.index.names)
        if index_length == 1:
            row_total = pd.DataFrame(totals, index=[col_row_name])
            df = pd.concat([df, row_total], axis=0, ignore_index=ignore_index)
            if ignore_index:
                # Place row total name column before first numeric column
                first_col = df.select_dtypes(include='number').columns[0]
                first_pos = df.columns.get_loc(first_col)
                if first_pos >= 1:
                    tot_colpos = first_pos - 1
                    df.iloc[-1, tot_colpos] = col_row_name
        else:
            total_row = ['' for _ in df.index.names]
            total_row[index_length - 1] = col_row_name
            index = tuple(total_row)

            for col, total in totals.items():
                df.loc[index, col] = total

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


# assign() {{{1
def assign(df: pd.DataFrame,
           *args,
           **kwargs) -> pd.DataFrame:
    '''Assign new columns to a DataFrame.

    Returns a new object with all original columns in addition to new ones.
    Existing columns that are re-assigned will be overwritten.

    Parameters
    ----------
    df
        pandas dataframe
    args
        arbitrary arguments (for future use)
    **kwargs
        dict of {str: callable or Series}

        The column names are keywords. If the values are callable, they are
        computed on the DataFrame and assigned to the new columns. The callable
        must not change input DataFrame (though pandas doesn't check it). If the
        values are not callable, (e.g. a Series, scalar, or array), they are
        simply assigned.

        .. note::

            If you wish to apply a function to a columns set of values:
            pass a tuple with column name and function to call.
            For example:

        .. code-block::

            assign(reversed=('regions', lambda x: x[::-1]))

        is converted to:

        .. code-block::

            assign(reversed=lambda x: x['regions'].apply(lambda x: x[::-1]))

    Returns
    -------
    A pandas DataFrame
        A new DataFrame with the new columns in addition to
        all the existing columns.

    Notes
    -----
    Assigning multiple columns within the same ``assign`` is possible.
    Later items in '\\*\\*kwargs' may refer to newly created or modified
    columns in 'df'; items are computed and assigned into 'df' in order.

    Examples
    --------
    You can create a dictionary of column names to corresponding functions,
    then pass the dictionary to the assign function as shown below:

    .. code-block::

        %%piper --dot

        sample_data()
        .. assign(**{'reversed': ('regions', lambda x: x[::-1]),
                     'v1_x_10': lambda x: x.values_1 * 10,
                     'v2_div_4': lambda x: x.values_2 / 4,
                     'dow': lambda x: x.dates.dt.day_name(),
                     'ref': lambda x: x.v2_div_4 * 5,
                     'ids': lambda x: x.ids.astype('category')})
        .. across(['values_1', 'values_2'], lambda x: x.astype(float),
                  series_obj=True)
        .. relocate('dow', 'after', 'dates')
        .. select(['-dates', '-order_dates'])
        .. head(tablefmt='plain')

            dow       countries regions  ids values_1 values_2 reversed v1_x_10 v2_div_4  ref
         0  Wednesday Italy     East     A        311       26 tsaE        3110        6   32
         1  Thursday  Portugal  South    D        150      375 htuoS       1500       94  469
         2  Friday    Spain     East     A        396       88 tsaE        3960       22  110
         3  Saturday  Italy     East     B        319      233 tsaE        3190       58  291

    .. code-block::

        %%piper

        sample_sales()
        >> select()
        >> assign(month_plus_one = lambda x: x.month + pd.Timedelta(1, 'D'),
                  alt_formula = lambda x: x.actual_sales * .2)
        >> select(['-target_profit'])
        >> assign(profit_sales = lambda x: x.actual_profit - x.actual_sales,
                  month_value = lambda x: x.month.dt.month,
                  product_added = lambda x: x['product'] + 'TEST',
                  salesthing = lambda x: x.target_sales.sum())
        >> select(['-month', '-actual_sales', '-actual_profit', '-month_value', '-location'])
        >> assign(profit_sales = lambda x: x.profit_sales.astype(int))
        >> reset_index(drop=True)
        >> head(tablefmt='plain')

            product   target_sales month_plus_one alt_formula profit_sales product_added salesthing
         0  Beachwear        31749     2021-01-02        5842       -27456 BeachwearTEST    5423715
         1  Beachwear        37833     2021-01-02        6810       -28601 BeachwearTEST    5423715
         2  Jeans            29485     2021-01-02        6310       -27132 JeansTEST        5423715
         3  Jeans            37524     2021-01-02        8180       -36811 JeansTEST        5423715

    '''
    if kwargs:
        for keyword, value in kwargs.items():
            if isinstance(value, tuple):
                column_to_apply_function, function = value
                new_function = lambda x: x[column_to_apply_function].apply(function)
                kwargs[keyword] = new_function

    try:
        df = df.assign(*args, **kwargs)
    except ValueError as e:
        logger.info(e)
        raise ValueError('''Try assign(column='column_to_apply_func', function)''')

    return df


# columns() {{{1
def columns(df: pd.DataFrame,
            regex: str = None,
            astype: str = 'list') -> Union[str, list, dict, pd.Series, pd.DataFrame]:
    '''show dataframe column information

    This function is useful reviewing or manipulating column(s)

    The dictionary output is particularly useful when composing the rename of
    multiple columns.

    Examples
    --------

    .. code-block::

        import numpy as np
        import pandas as pd

        id_list = ['A', 'B', 'C', 'D', 'E']
        s1 = pd.Series(np.random.choice(id_list, size=5), name='ids')

        region_list = ['East', 'West', 'North', 'South']
        s2 = pd.Series(np.random.choice(region_list, size=5), name='regions')

        df = pd.concat([s1, s2], axis=1)

        columns(df, 'list')

        ['ids', 'regions']


    Parameters
    ----------
    df
        dataframe
    regex
        Default None. regular expression to 'filter' list of returned columns.
    astype
        Default 'list'. See return options below:
            - 'dict' returns a dictionary object
            - 'list' returns a list object
            - 'text' returns columns joined into a text string
            - 'series' returns a pd.Series
            - 'dataframe' returns a pd.DataFrame


    Returns
    -------
    dictionary, list, str, pd.Series, pd.DataFrame
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

    # note: mypy needs to see this 'dummy' catch all return statement
    return None


# count() {{{1
def count(df: pd.DataFrame,
          columns: Union[str, list] = None,
          totals_name: str = 'n',
          percent: bool = True,
          cum_percent: bool = True,
          threshold: int = 100,
          round: int = 2,
          totals: bool = False,
          sort_values: bool = False,
          reset_index: bool = False):
    '''show column/category/factor frequency

    For selected column or multi-index show the frequency count, frequency %,
    cum frequency %. Also provides optional totals.

    Examples
    --------
    .. code-block::

        import numpy as np
        import pandas as pd
        from piper.defaults import *

        np.random.seed(42)

        id_list = ['A', 'B', 'C', 'D', 'E']
        s1 = pd.Series(np.random.choice(id_list, size=5), name='ids')
        s2 = pd.Series(np.random.randint(1, 10, s1.shape[0]), name='values')
        df = pd.concat([s1, s2], axis=1)

        %%piper
        count(df.ids, totals=True)
        >> head(tablefmt='plain')

                 n    %  cum %
        E        3   60  60.0
        C        1   20  80.0
        D        1   20  100.0
        Total    5  100


        %piper df >> count('ids', totals=False, cum_percent=False) >> head(tablefmt='plain')

        ids      n    %
        E        3   60
        C        1   20
        D        1   20


        %%piper
        df
        >> count(['ids'], sort_values=None, totals=True)
        >> head(tablefmt='plain')

                 n    %  cum %
        C        1   20  20.0
        D        1   20  40.0
        E        3   60  100.0
        Total    5  100

    Parameters
    ----------
    df
        dataframe reference
    columns
        dataframe columns/index to be used in groupby function
    totals_name
        name of total column, default 'n'
    percent
        provide % total, default True
    cum_percent
        provide cum % total, default True
    threshold
        filter cum_percent by this value, default 100
    round
        round decimals, default 2
    totals
        add total column, default False
    sort_values
        default False, None means use index sort
    reset_index
        default False


    Returns
    -------
    A pandas dataframe
    '''
    # NOTE:: pd.groupby by default does not count nans!
    try:
        if isinstance(columns, str):
            p1 = df.groupby(columns, dropna=False).agg(totals=(columns, lambda x: x.value_counts(dropna=False)))
        elif isinstance(df, pd.Series):
            new_df = df.to_frame()
            columns = new_df.columns.tolist()[0]
            p1 = new_df.groupby(columns, dropna=False).agg(totals=(columns, lambda x: x.value_counts(dropna=False)))
        elif isinstance(df, pd.DataFrame):
            if columns is not None:
                p1 = df.groupby(columns, dropna=False).agg(totals=(columns[0], lambda x: x.value_counts(dropna=False)))
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
                       ignore_index=reset_index)

    return p1


# clean_columns() {{{1
def clean_columns(df: pd.DataFrame,
                  replace_char: tuple = (' ', '_'),
                  title: bool = False) -> pd.DataFrame:
    '''Clean column names, strip blanks, lowercase, snake_case.

    Also removes awkward characters to allow for easier manipulation.
    (optionally 'title' each column name.)

    Examples
    --------
    .. code-block::

        column_list = [
            'dupe**', 'Customer   ', 'mdm no. to use', 'Target-name   ', '     Public',
            '_  Material', 'Prod type', '#Effective     ', 'Expired', 'Price%  ',
            'Currency$'
        ]

        df = pd.DataFrame(None, columns=column_list)
        df.columns.tolist()

        ['dupe**',
         'Customer   ',
         'mdm no. to use',
         'Target-name   ',
         '     Public',
         '_  Material',
         'Prod type',
         '#Effective     ',
         'Expired',
         'Price%  ',
         'Currency$']

    .. code-block::

        df = clean_columns(df)
        df.columns.tolist()

        ['dupe',
         'customer',
         'mdm_no_to_use',
         'target_name',
         'public',
         'material',
         'prod_type',
         'effective',
         'expired',
         'price',
         'currency']

    .. code-block::

        df = clean_columns(df, ('_', ' '), title=True)
        df.columns.tolist()

        ['Dupe',
         'Customer',
         'Mdm No To Use',
         'Target Name',
         'Public',
         'Material',
         'Prod Type',
         'Effective',
         'Expired',
         'Price',
         'Currency']


    Parameters
    ----------
    df
        pandas dataframe
    replace_char
        default (' ', '_'). A tuple giving the 'from' and 'to' characters to be replaced
    title
        default False. If True, titleize column values


    Returns
    -------
    pandas DataFrame object
    '''
    from_char, to_char = replace_char
    columns = [x.strip().lower() for x in df.columns]

    # Remove special chars at the beginning and end of column names
    special_chars = r'[\.\*\#\%\$\-\_]+'
    columns = [re.sub(f'^{special_chars}', '', x) for x in columns]
    columns = [re.sub(f'{special_chars}$', '', x) for x in columns]

    # Any embedded special characters in the middle of words, replace with a blank
    columns = [re.sub(f'{special_chars}', ' ', x) for x in columns]

    # All special chars should now be removed, except for to_char perhaps
    # and embedded spaces.
    columns = [re.sub(f'{to_char}+', ' ', x) for x in columns]

    # Strip and lowercase as default
    columns = [x.strip().lower() for x in columns]

    # Replace any remaining embedded blanks with single replacement 'to_char' value.
    columns = [re.sub('\s+', to_char, x) for x in columns]

    columns = [re.sub(from_char, to_char, x) for x in columns]

    if title:
        columns = [x.title() for x in columns]

    df.columns = columns

    return df


# combine_header_rows() {{{1
def combine_header_rows(df: pd.DataFrame,
                        start: int = 0,
                        end: int = 1,
                        delimitter: str = ' ',
                        title: bool = True,
                        infer_objects: bool = True) -> pd.DataFrame:
    '''Combine 1 or more header rows across into one.

    Optionally, infers remaining data column data types.

    Examples
    --------
    .. code-block::

        data = {'A': ['Customer', 'id', 48015346, 49512432],
                'B': ['Order', 'Number', 'DE-12345', 'FR-12346'],
                'C': [np.nan, 'Qty', 10, 40],
                'D': ['Item', 'Number', 'SW-10-2134', 'YH-22-2030'],
                'E': [np.nan, 'Description', 'Screwdriver Set', 'Workbench']}

        df = pd.DataFrame(data)
        head(df, tablefmt='plain')

            A         B         C    D           E
         0  Customer  Order     nan  Item        nan
         1  id        Number    Qty  Number      Description
         2  48015346  DE-12345  10   SW-10-2134  Screwdriver Set
         3  49512432  FR-12346  40   YH-22-2030  Workbench

    .. code-block::

        df.iloc[0] = df.iloc[0].ffill()
        df = combine_header_rows(df)
        head(df, tablefmt='plain')

              Customer Id  Order Number      Order Qty  Item Number    Item Description
         2       48015346  DE-12345                 10  SW-10-2134     Screwdriver Set
         3       49512432  FR-12346                 40  YH-22-2030     Workbench


    Parameters
    ----------
    df
        dataframe
    start
        starting row - default 0
    end
        ending row to combine, - default 1
    delimitter
        character to be used to 'join' row values together. default is ' '
    title
        default False. If True, titleize column values
    infer_objects
        default True. Infer data type of resultant dataframe


    Returns
    -------
    A pandas dataframe
    '''
    data = df.iloc[start].astype(str).values
    rows = range(start+1, end+1)

    for row in rows:
        data = data + delimitter + df.iloc[row].astype(str).values

    df.columns = data

    # Read remaining data and reference back to dataframe reference
    df = df.iloc[end + 1:]

    if infer_objects:
        df = df.infer_objects()

    df = clean_columns(df, replace_char=(' ', '_'), title=title)
    df = clean_columns(df, replace_char=('_', ' '), title=title)

    return df


# distinct() {{{1
def distinct(df: pd.DataFrame,
             *args,
             shape: bool = False,
             **kwargs) -> pd.DataFrame:
    '''select distinct/unique rows

    This is a wrapper function rather than using e.g. df.distinct()
    For details of args, kwargs - see help(pd.DataFrame.distinct)

    Examples
    --------
    .. code-block::

        from piper.verbs import select, distinct
        from piper.factory import sample_data

        df = sample_data()
        df = select(df, ['countries', 'regions', 'ids'])
        df = distinct(df, 'ids', shape=False)

        expected = (5, 3)
        actual = df.shape


    Parameters
    ----------
    df
        dataframe
    shape
        default True. Show shape information as a logger.info() message
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    result = df.drop_duplicates(*args, **kwargs)

    if shape:
        _shape(df)

    return result


# drop() {{{1
def drop(df: pd.DataFrame,
         *args,
         **kwargs) -> pd.DataFrame:
    '''drop column(s)

    This is a wrapper function rather than using e.g. df.drop()
    For details of args, kwargs - see help(pd.DataFrame.drop)

    Examples
    --------
    .. code-block::

        df <- pd.read_csv('inputs/export.dsv', sep='\t')
        >> clean_columns()
        >> trim()
        >> assign(adast = lambda x: x.adast.astype('category'),
                  adcrcd = lambda x: x.adcrcd.astype('category'),
                  aduom = lambda x: x.aduom.astype('category'))
        >> drop(columns='adcrcd_1')


    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    return df.drop(*args, **kwargs)


# drop_columns() {{{1
def drop_columns(df: pd.DataFrame,
                 value: Union[str, int, float] = None,
                 how: str = 'all') -> pd.DataFrame:
    ''' drop columns containing blanks, zeros or na

    Examples
    --------
    .. code-block::

        %%piper
        dummy_dataframe()
        >> drop_columns()

    Parameters
    ----------
    df
        pandas dataframe
    value
        Default is None. Drop a column if it contains a blank value in every row.
        Enter a literal string or numeric value to check against.

        Special value - 'isna'. If specified, depending on the 'how' parameter
        drop columns containing na / null values.
    how
        {None, 'any', 'all'}, default 'all'.
        Determine if row or column is removed from DataFrame, when we have
        at least one NA or all NA.

        * 'any' : If any NA values are present, drop that row or column.
        * 'all' : If all values are NA, drop that row or column.

    Returns
    -------
    A pandas DataFrame
    '''
    if value == 'isna':
        df = df.dropna(axis=1, how=how)
    else:
        if value is None:
            value = ""

        for col in df.columns:
            drop_column = np.where(df[col] == value, 1, 0).sum() == df.shape[0]
            if drop_column:
                df = df.drop(columns=col, axis=1)

    return df


# duplicated() {{{1
def duplicated(df: pd.DataFrame,
               subset: Union[str, List[str]] = None,
               keep: bool = False,
               sort: bool = True,
               column: str = 'duplicate',
               loc: str = 'first',
               ref_column: str = None,
               duplicates: bool = False) -> pd.DataFrame:
    '''locate duplicate data

    .. note::

        Returns a copy of the input dataframe object

    Examples
    --------
    .. code-block::

        from piper.factory import simple_series
        df = simple_series().to_frame()

        df = duplicated(df, keep='first', sort=True)
        df

            ids    duplicate
         2  C      False
         0  D      False
         1  E      False
         3  E      True

    Parameters
    ----------
    df
        pandas dataframe
    subset
        column label or sequence of labels, required
        Only consider certain columns for identifying duplicates
        Default None - consider ALL dataframe columns
    keep
        {‘first’, ‘last’, False}, default ‘first’
        first : Mark duplicates as True except for the first occurrence.
        last : Mark duplicates as True except for the last occurrence.
        False : Mark all duplicates as True.
    sort
        If True sort returned dataframe using subset fields as key
    column
        Insert a column name identifying whether duplicate
        (True/False), default 'duplicate'
    duplicates
        Default True. Return only duplicate key rows


    Returns
    -------
    pandas dataframe
    '''
    df = _dataframe_copy(df, inplace=False)

    if subset is None:
        subset = df.columns.tolist()

    df[column] = df.duplicated(subset, keep=keep)

    if duplicates:
        df = df[df[column]]

    if sort:
        df = df.sort_values(subset)

    return df


# explode() {{{1
def explode(df: pd.DataFrame,
            *args,
            **kwargs) -> pd.DataFrame:
    '''Transform list-like column values to rows

    This is a wrapper function rather than using e.g. df.explode()
    For details of args, kwargs - see help(pd.DataFrame.explode)

    Examples
    --------

    .. code-block::

        from piper.factory import sample_data

        df = sample_data()
        df = group_by(df, 'countries')
        df = summarise(df, ids=('ids', set))
        df.head()

          countries     ids
          Italy         {'B', 'C', 'D', 'A', 'E'}
          Portugal      {'B', 'C', 'A', 'D', 'E'}
          Spain         {'B', 'C', 'D', 'A', 'E'}
          Switzerland   {'B', 'C', 'A', 'D', 'E'}
          Sweden        {'B', 'C', 'A', 'D', 'E'}

    .. code-block::

        explode(df, 'ids').head(8)

          countries     ids
          Italy         B
          Italy         C
          Italy         D
          Italy         A
          Italy         E
          Portugal      B
          Portugal      C
          Portugal      A

    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''

    return df.explode(*args, **kwargs)


# flatten_cols() {{{1
def flatten_cols(df: pd.DataFrame,
                 join_char: str = '_',
                 remove_prefix=None) -> pd.DataFrame:
    '''Flatten multi-index column headings

    Examples
    --------
    .. code-block::

        from piper.defaults import *

        %%piper

        sample_data()
        >> group_by(['countries', 'regions'])
        >> summarise(totalval1=('values_1', 'sum'))
        >> assign(mike='x.totalval1 * 50', eoin='x.totalval1 * 100')
        >> transform()
        >> order_by(['countries', '-g%'])
        >> unstack()
        >> flatten_cols(remove_prefix='mike|eoin|totalval1')
        >> reset_index()
        >> set_index('countries')
        >> head()


    Parameters
    ----------
    df
        pd.DataFrame - dataframe
    join_char
        delimitter joining 'prefix' value(s) to be removed.
    remove_prefix
        string(s) (delimitted by pipe '|')
        e.g. ='mike|eoin|totalval1'


    Returns
    -------
    A pandas dataframe
    '''
    def _flatten(column_string, join_char='_', remove_prefix=None):
        ''' Takes a dataframe column string value.
        if tuple (from a groupby/stack/pivot df) returns a 'joined' string
        otherwise returns the original string.
        '''
        modified_string = column_string

        if isinstance(modified_string, tuple):
            modified_string = join_char.join([str(x) for x in modified_string]).strip(join_char)

        if isinstance(modified_string, str):
            if remove_prefix:
                modified_string = re.sub(remove_prefix, '', modified_string).strip(join_char)

        return modified_string

    df.columns = [_flatten(x, join_char, remove_prefix) for x in df.columns]

    return df


# fmt_dateidx() {{{1
def fmt_dateidx(df: pd.DataFrame,
                freq: str = 'D') -> pd.DataFrame:
    '''format dataframe datelike index

    Checks if dataframe index contains a date-like grouper object.
    If so, applies the 'frequency' string given.

    Examples
    --------

    Parameters
    ----------
    df
        dataframe
    freq
        Default 'd' (days)
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        for list of valid frequency strings.


    Returns
    -------
    A pandas DataFrame
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
def group_by(df: pd.DataFrame,
             *args,
             freq: str = 'D',
             **kwargs) -> pd.DataFrame.groupby:
    '''Group by dataframe

    This is a wrapper function rather than using e.g. df.groupby()
    For details of args, kwargs - see help(pd.DataFrame.groupby)

    The first argument or by keyword are checked in turn and converted
    to pd.Grouper objects. If any group fields are 'date' like, they
    can be represented as various 'frequencies' types.

    See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    for list of valid frequency strings.


    Examples
    --------

    .. code-block::

        %%piper
        sample_data()
        >> where("ids == 'A'")
        >> where("values_1 > 300 & countries.isin(['Italy', 'Spain'])")
        >> group_by(['countries', 'regions'])
        >> summarise(total_values_1=pd.NamedAgg('values_1', 'sum'))


    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    freq
        Default 'd' (days)
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    pandas DataFrame object
    '''
    args_copy = list(args)
    logger.debug(args_copy)
    logger.debug(kwargs)

    if kwargs.get('by') != None:
        index = _set_grouper(df, kwargs.get('by'), freq=freq)
        kwargs['by'] = index

    if len(args) > 0:
        args_copy[0] = _set_grouper(df, args[0], freq=freq)

    df = df.groupby(*args_copy, **kwargs)

    return df


# head() {{{1
def head(df: pd.DataFrame,
         n: int = 4,
         shape: bool = True,
         tablefmt: bool = False,
         precision: int = 0) -> pd.DataFrame:
    '''show first n records of a dataframe.

    Like the corresponding R function, displays the first n records.
    Alternative to using df.head()

    Examples
    --------
    .. code-block::

        head(df)


    Parameters
    ----------
    df
        pandas dataframe
    n
        Default n=4. number of rows to display
    shape
        Default True. Show shape information
    tablefmt
        Default False. If supplied, tablefmt keyword value can be any valid
        format supplied by the tabulate package
    precision
        Default 0. Number precision shown if tablefmt specified

    Returns
    -------
    A pandas dataframe
    '''
    if shape:
        _shape(df)

    if tablefmt:
        print(df.head(n=n).to_markdown(tablefmt=tablefmt, floatfmt=f".{precision}f"))
        return

    return df.head(n=n)


# info() {{{1
def info(df,
         n_dupes: bool = False,
         fillna: bool = False,
         memory_info: bool = True) -> pd.DataFrame:
    '''show dataframe meta data

    Provides a summary of the structure of the dataframe data.

    Examples
    --------
    .. code-block::

        import numpy as np
        import pandas as pd
        from piper.verbs import info

        np.random.seed(42)

        id_list = ['A', 'B', 'C', 'D', 'E']
        s1 = pd.Series(np.random.choice(id_list, size=5), name='ids')
        s2 = pd.Series(np.random.randint(1, 10, s1.shape[0]), name='values')
        df = pd.concat([s1, s2], axis=1)

               columns     type       n     isna     isnull     unique
           0   ids         object     5        0          0          3
           1   values      int64      5        0          0          4


    Parameters
    ----------
    df
        a pandas dataframe
    n_dupes
        default False. Column showing the numbers of groups of duplicate records
    fillna
        default: False. Filter out only columns which contain nulls/np.nan
    memory
        If True, show dataframe memory consumption in mb


    Returns
    -------
    A pandas dataframe
    '''
    null_list = [df[col].isnull().sum() for col in df]
    na_list = [df[col].isna().sum() for col in df]
    dtypes_list = [df[col].dtype for col in df]
    nunique_list = [df[col].nunique() for col in df]
    dupes_list = [df.duplicated(col, keep=False).sum() for col in df]
    total_list = [df[col].value_counts(dropna=False).sum() for col in df]
    infer_list = [pd.api.types.infer_dtype(df[col]) for col in df]

    dicta = {'columns': df.columns.values, 'type': dtypes_list,
             'inferred': infer_list, 'n': total_list, 'isna': na_list,
             'isnull': null_list, 'unique': nunique_list}

    if n_dupes:
        dicta.update({'n dupes': dupes_list})

    dr = pd.DataFrame.from_dict(dicta)

    if fillna:
        return dr.query('isna > 0')

    if memory_info:
        memory(df)

    return dr


# inner_join() {{{1
def inner_join(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    '''df (All) | df2 (All) matching records only

    This is a wrapper function rather than using e.g. df.merge(how='inner')
    For details of args, kwargs - see help(pd.DataFrame.merge)

    Examples
    --------
    .. code-block::

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


    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    kwargs['how'] = 'inner'
    logger.debug(f"{kwargs}")

    return df.merge(*args, **kwargs)


# left_join() {{{1
def left_join(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    '''df (All) | df2 (All/na) df always returned

    This is a wrapper function rather than using e.g. df.merge(how='left')
    For details of args, kwargs - see help(pd.DataFrame.merge)

    Examples
    --------
    .. code-block::

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


# memory() {{{1
def memory(df: pd.DataFrame) -> pd.DataFrame:
    '''show dataframe consumed memory (mb)

    Examples
    --------
    .. code-block::

        from piper.factory import sample_sales
        from piper.verbs import memory
        import piper
        memory(sample_sales())
        >> Dataframe consumes 0.03 Mb

    Parameters
    ----------
    df
        pandas dataframe

    Returns
    -------
    A pandas dataframe
    '''
    memory = df.memory_usage(deep=True).sum()
    memory = round(memory / (1024 * 1024), 2)

    msg = f'Dataframe consumes {memory} Mb'
    logger.info(msg)

    return df


# non_alpha() {{{1
def non_alpha(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    '''check for non-alphanumeric characters

    Parameters
    ----------
    df
        pandas dataframe
    col_name
        name of column to be checked


    Returns
    -------
    pandas dataframe with additional column containing True/False
    tests of the selected column having special characters

    '''
    pattern = r'[a-zA-Z0-9]*$'

    df['non_alpha'] = ~df[col_name].str.match(pattern)

    return df


# order_by() {{{1
def order_by(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    '''order/sequence dataframe

    @__TODO__ Review mypy validation rules and update if required

    This is a wrapper function rather than using e.g. df.sort_values()
    For details of args, kwargs - see help(pd.DataFrame.sort_values)

    The first argument and/or keyword 'by' is checked to see:

    - For each specified column value
      If it starts with a minus sign, assume ascending=False

    Examples
    --------
    .. code-block::

        %%piper

        sample_data()
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


    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    # args is a tuple, therefore needs to be cast to a list for manipulation
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

        f = lambda ix, x: column_seq[ix] if x else column_seq[ix][1:] #type: ignore
        if args_copy != []:
            args_copy[0] = [f(ix, x) for ix, x in enumerate(sort_seq)] #type: ignore
        else:
            kwargs['by'] = [f(ix, x) for ix, x in enumerate(sort_seq)] #type: ignore

    else:
        # Assume ascending sequence, unless otherwise specified
        if kwargs.get('ascending') is None:

            kwargs['ascending'] = f(column_seq) #type: ignore
            if kwargs['ascending'] == False:
                column_seq = column_seq[1:] #type: ignore

            if args_copy != []:
                args_copy[0] = column_seq
            else:
                kwargs['by'] = column_seq

    return df.sort_values(*args_copy, **kwargs)


# outer_join() {{{1
def outer_join(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    '''df (All/na) | df2 (All/na) All rows returned

    This is a wrapper function rather than using e.g. df.merge(how='outer')
    For details of args, kwargs - see help(pd.DataFrame.merge)

    Examples
    --------
    .. code-block::

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

    .. code-block::

            OrderNo   Status     Type_     description      description_2
               1001   A          SO        Standing Order   another one
               1003   A          SO        Standing Order   another one
               1002   C          SA        Sales Order      Arbitrary desc
               1004   A          DA        nan              nan
               1005   P          DD        nan              nan
    '''
    kwargs['how'] = 'outer'
    logger.debug(f"{kwargs}")

    return df.merge(*args, **kwargs)


# overlaps() {{{1
def overlaps(df: pd.DataFrame,
             unique_key: Union[str, List[str]] = None,
             start: str = 'effective',
             end: str = 'expiry',
             overlaps: str = 'overlaps') -> pd.DataFrame:
    '''Analyse dataframe rows with overlapping date periods

    Examples
    --------
    .. code-block::

        data = {'prices': [100, 200, 300],
                'contract': ['A', 'B', 'A'],
                'effective': ['2020-01-01', '2020-03-03', '2020-05-30'],
                'expired': ['2020-12-31', '2021-04-30', '2022-04-01']}

        df = pd.DataFrame(data)
        overlaps(df, start='effective', end='expired', unique_key='contract')

                 prices   contract     effective     expired      overlaps
           0        100   A            2020-01-01    2020-12-31   True
           1        200   B            2020-03-03    2021-04-30   False
           2        300   A            2020-05-30    2022-04-01   True


    Parameters
    ----------
    df
        dataframe
    unique_key
        column(s) that uniquely identify rows
    start
        column that defines start/effective date, default 'effective_date'
    end
        column that defines end/expired date, default 'expiry_date'
    overlaps
        default 'overlaps'. Name of overlapping column containing True/False values


    Returns
    -------
    pandas dataframe with boolean based overlap column
    '''
    if unique_key is None:
        raise ValueError('Please provide unique key.')

    if isinstance(unique_key, str):
        unique_key = [unique_key]

    key_cols = unique_key + [start, end]

    missing_cols = ', '.join([col for col in key_cols if col not in df.columns.tolist()])
    if len(missing_cols) > 0:
        raise KeyError(f"'{missing_cols}' not found in dataframe")

    dfa = df[key_cols]
    dfa.insert(0, 'index', df.index)
    dfb = dfa

    merged = dfa.merge(dfb, how='left', on=unique_key)

    criteria_1 = merged[f'{end}_x'] >= merged[f'{start}_y']
    criteria_2 = merged[f'{start}_x'] <= merged[f'{end}_y']
    criteria_3 = merged.index_x != merged.index_y

    merged[overlaps] = criteria_1 & criteria_2 & criteria_3

    # Find unique keys with overlapping dates
    merged[overlaps] = criteria_1 & criteria_2 & criteria_3
    cols = unique_key + [overlaps]

    # We need just the unique key and whether it overlaps or not
    merged = merged[cols][merged[overlaps]].drop_duplicates()

    merged = df.merge(merged, how='left', on=unique_key)
    merged[overlaps] = merged[overlaps].fillna(False)

    return merged


# pivot_longer() {{{1
def pivot_longer(df: pd.DataFrame,
          *args,
          **kwargs) -> pd.DataFrame:
    '''pivot dataframe wide to long

    This is a wrapper function rather than using e.g. df.melt()
    For details of args, kwargs - see help(pd.DataFrame.melt)


    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    return df.melt(*args, **kwargs)


# pivot_table() {{{1
def pivot_table(df: pd.DataFrame,
                *args,
                freq: str = 'M',
                format_date: bool = False,
                **kwargs) -> pd.DataFrame:
    '''create Excel like pivot table

    This is a wrapper function rather than using e.g. df.pivot_table()
    For details of args, kwargs - see help(pd.DataFrame.pivot_table)

    Examples
    --------
    .. code-block::

        from piper.verbs import pivot_wider
        from piper.factory import sample_data
        import piper.defaults

        df = sample_data()

        index=['dates', 'order_dates', 'regions', 'ids']
        pvt = pivot_wider(df, index=index, freq='Q', format_date=True)
        pvt.head()

        |                                       |   values_1 |   values_2 |
        |:--------------------------------------|-----------:|-----------:|
        | ('Mar 2020', 'Mar 2020', 'East', 'A') |    227.875 |     184.25 |
        | ('Mar 2020', 'Mar 2020', 'East', 'B') |    203.2   |     168    |
        | ('Mar 2020', 'Mar 2020', 'East', 'C') |    126     |     367    |
        | ('Mar 2020', 'Mar 2020', 'East', 'D') |    125.667 |     259    |
        | ('Mar 2020', 'Mar 2020', 'East', 'E') |    194     |     219    |


    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    pandas DataFrame object
    '''
    logger.debug(args)
    logger.debug(kwargs)

    if kwargs.get('index') != None:
        index = _set_grouper(df, kwargs.get('index'), freq=freq)
        kwargs['index'] = index

    df = pd.pivot_table(df, *args, **kwargs)

    if format_date:
        df = fmt_dateidx(df, freq=freq)

    return df


# relocate() {{{1
def relocate(df: pd.DataFrame,
             column: Union[str, list] = None,
             loc: str = 'last',
             ref_column: str = None,
             index: bool = False) -> pd.DataFrame:
    '''move column(s) in a dataframe

    Based on the corresponding R function - relocate

    Examples
    --------
    .. code-block::

        %%piper
        sample_matrix()
        >> relocate('e', 'before', 'b')
        >> select('-c')
        >> where("a < 20 and b > 1")
        >> order_by(['-d'])
        >> head(5, tablefmt='plain')

              a    e    b    d
         0   15    8    9   25
         1    8   15   26    5
         2    5   -7    5   -9

    .. code-block::

        %%piper
        sample_sales()
        >> select(('location', 'target_profit'))
        >> pd.DataFrame.set_index(['location', 'product'])
        >> head(tablefmt='plain')

                                 month                  target_sales    target_profit
        ('London', 'Beachwear')  2021-01-01 00:00:00           31749             1905
        ('London', 'Beachwear')  2021-01-01 00:00:00           37833             6053
        ('London', 'Jeans')      2021-01-01 00:00:00           29485             4128
        ('London', 'Jeans')      2021-01-01 00:00:00           37524             3752

        %%piper
        sample_sales()
        >> select(('location', 'target_profit'))
        >> pd.DataFrame.set_index(['location', 'product'])
        >> relocate(column='location', loc='after', ref_column='product', index=True)
        >> head(tablefmt='plain')

                                 month                  target_sales    target_profit
        ('Beachwear', 'London')  2021-01-01 00:00:00           31749             1905
        ('Beachwear', 'London')  2021-01-01 00:00:00           37833             6053
        ('Jeans', 'London')      2021-01-01 00:00:00           29485             4128
        ('Jeans', 'London')      2021-01-01 00:00:00           37524             3752


    **NOTE**
    If you omit the keyword parameters, it probably 'reads' better ;)
    >> relocate(location', 'after', 'product', index=True)


    Parameters
    ----------
    df
        dataframe
    column
        column name(s) to be moved
    loc
        default -'last'. The relative location to move the column(s) to.
        Valid values are: 'first', 'last', 'before', 'after'.
    ref_column
        Default - None. Reference column
    index
        default is False (column). If True, then the column(s) being moved are
        considered to be row indexes.


    Returns
    -------
    A pandas dataframe
    '''
    def sequence(columns, index=False):
        ''' Return column or index sequence '''

        if index:
            return df.reorder_levels(columns)
        else:
            return df[columns]

    if column is None:
        raise KeyError(f'Please enter column(s) to be relocated/moved.')

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


# rename() {{{1
def rename(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame :
    '''rename dataframe col(s)

    This is a wrapper function rather than using e.g. df.rename()
    For details of args, kwargs - see help(pd.DataFrame.rename)

    Examples
    --------
    .. code-block::

        %%piper
        sample_sales()
        >> rename(columns={'product': 'item'})


    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    pandas DataFrame object
    '''
    return df.rename(*args, **kwargs)


# rename_axis() {{{1
def rename_axis(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame :
    '''rename dataframe axis

    This is a wrapper function rather than using e.g. df.rename_axis()
    For details of args, kwargs - see help(pd.DataFrame.rename_axis)

    Examples
    --------
    .. code-block::

        %%piper
        sample_sales()
        >> pivot_wider(index=['location', 'product'], values='target_sales')
        >> reset_index()
        >> head(tablefmt='plain')

            location    product       target_sales
         0  London      Beachwear            25478
         1  London      Footwear             25055
         2  London      Jeans                25906
         3  London      Sportswear           26671

        %%piper
        sample_sales()
        >> pivot_wider(index=['location', 'product'], values='target_sales')
        >> rename_axis(('AAA', 'BBB'), axis='rows')
        >> head(tablefmt='plain')

            AAA     BBB           target_sales
         0  London  Beachwear            25478
         1  London  Footwear             25055
         2  London  Jeans                25906
         3  London  Sportswear           26671


    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    pandas DataFrame object

    '''
    return df.rename_axis(*args, **kwargs)


# reset_index() {{{1
def reset_index(df: pd.DataFrame,
          *args,
          **kwargs) -> pd.DataFrame:
    '''reset_index dataframe

    This is a wrapper function rather than using e.g. df.reset_index()
    For details of args, kwargs - see help(pd.DataFrame.reset_index)


    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    return df.reset_index(*args, **kwargs)


# right_join() {{{1
def right_join(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    '''df (All/na) | df2 (All) df2 always returned

    This is a wrapper function rather than using e.g. df.merge(how='right')
    For details of args, kwargs - see help(pd.DataFrame.merge)

    Examples
    --------
    .. code-block::

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

            OrderNo   Status     Type_     description      description_2
               1002   C          SA        Sales Order      Arbitrary desc
               1001   A          SO        Standing Order   another one
               1003   A          SO        Standing Order   another one

    '''
    kwargs['how'] = 'right'
    logger.debug(f"{kwargs}")

    return df.merge(*args, **kwargs)


# sample() {{{1
def sample(df: pd.DataFrame,
           n: int = 2,
           shape: bool = True,
           *args,
           **kwargs):
    '''show sample data

    This is a wrapper function rather than using e.g. df.sample()
    For details of args, kwargs - see help(pd.DataFrame.sample)

    Examples
    --------
    .. code-block::

        sample(df)


    Parameters
    ----------
    df
        dataframe
    shape
        show shape information as a logger.info() message
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    A pandas dataframe

    '''
    if info:
        _shape(df)

    if isinstance(df, pd.Series):
        return df.sample(n=n, *args, **kwargs).to_frame()

    return df.sample(n=n, *args, **kwargs)


# select() {{{1
def select(df: pd.DataFrame,
           *args,
           regex: str = None,
           like: str = None,
           include: str = None,
           exclude: str = None) -> pd.DataFrame:
    '''select dataframe columns

    Inspired by the select() function from R tidyverse.

    Select column names from a dataframe

    Examples
    --------

    .. code-block::

        df = sample_sales()

        select(df) # select ALL columns

        # select column column listed
        select(df, 'location')

        # select columns listed
        select(df, ['product', 'target_sales'])

        # select ALL columns EXCEPT the column listed (identified by - minus sign prefix)
        select(df, '-target_profit')

        # select ALL columns EXCEPT the column specified with a minus sign within the list
        select(df, ['-target_sales', '-target_profit'])

        # select column range using a tuple from column up to and including the 'to' column.
        select(df, ('month', 'actual_profit'))

        # select all number data types
        select(df, '-month', include='number')

        # exclude all float data types including month column
        select(df, '-month', exclude='float')

        # select using a regex string
        select(df, regex='sales') -> select fields containing 'sales'
        select(df, regex='^sales') -> select fields starting with 'sales'
        select(df, regex='sales$') -> select fields ending with 'sales'

    .. note ::

        See the numpy dtype hierarchy. To select:
            - strings use the 'object' dtype, but note that this will return
              all object dtype columns
            - all numeric types, use np.number or 'number'
            - datetimes, use np.datetime64, 'datetime' or 'datetime64'
            - timedeltas, use np.timedelta64, 'timedelta' or 'timedelta64'
            - Pandas categorical dtypes, use 'category'
            - Pandas datetimetz dtypes, use 'datetimetz' (new in 0.20.0) or 'datetime64[ns, tz]'


    Parameters
    ----------
    df
        dataframe
    args
        if not specified, then ALL columns are selected
            - str   : single column (in quotes)
            - list  : list of column names (in quotes)
            - tuple : specify a range of column names or column positions (1-based)

        .. note::

            Prefixing column name with a minus sign filters out the column
            from the returned list of columns e.g. '-sales', '-month'

    regex
        Default None. Wrapper for regex keyword in pd.DataFrame.filter()
        Keep labels from axis for which re.search(regex, label) == True.
    like
        Default None. Wrapper for like keyword in pd.DataFrame.filter()
        Keep labels from axis for which like in label == True.
    include
        Default None. Wrapper for include keyword in pd.DataFrame.select_dtypes()
    exclude
        Default None. Wrapper for exclude keyword in pd.DataFrame.select_dtypes()


    Returns
    -------
    pandas DataFrame object
    '''
    columns = list(df.columns)
    selected: List = []
    drop: List = []

    for column_arg in args:
        if isinstance(column_arg, str):
            selected, drop = _check_col(column_arg, selected, drop, columns)

        # Tuples used to specify a 'range' of columns (from/to)
        if isinstance(column_arg, tuple):

            if sum([isinstance(v, str) for v in column_arg]) == 2:
                cols = list(df.loc[:, slice(*column_arg)].columns)
                for col in cols:
                    selected, drop = _check_col(col, selected, drop, columns)

            if sum([isinstance(v, int) for v in column_arg]) == 2:
                first, last = column_arg
                cols = list(df.iloc[:, range(first-1, last)].columns)
                for col in cols:
                    selected, drop = _check_col(col, selected, drop, columns)

        # Lists to be used for passing in a set of distinct values
        if isinstance(column_arg, list):
            for col in column_arg:
                selected, drop = _check_col(col, selected, drop, columns)

    if like is not None:
        cols = df.filter(like=like)
        for col in cols:
             selected, drop = _check_col(col, selected, drop, columns)

    if regex is not None:
        cols = df.filter(regex=regex)
        for col in cols:
             selected, drop = _check_col(col, selected, drop, columns)

    if include is not None:
        cols = df.select_dtypes(include=include)
        for col in cols:
            selected, drop = _check_col(col, selected, drop, columns)

    if exclude is not None:
        cols = df.select_dtypes(exclude=exclude)
        for col in cols:
            selected, drop = _check_col(col, selected, drop, columns)

    if selected == []:
        selected = columns

    if drop != []:
        for col in drop:
            if col in selected:
                selected.remove(col)

    return df[selected]

# set_columns() {{{1
def set_columns(df: pd.DataFrame,
                columns: Union[str, Any] = None) -> pd.DataFrame:
    '''set dataframe column names

    Parameters
    ----------
    df
        dataframe
    columns
        column(s) values to be changed in referenced dataframe


    Returns
    -------
    A pandas dataframe
    '''
    df.columns = columns

    return df


# set_index() {{{1
def set_index(df: pd.DataFrame,
          *args,
          **kwargs) -> pd.DataFrame:
    '''set_index dataframe

    This is a wrapper function rather than using e.g. df.set_index()
    For details of args, kwargs - see help(pd.DataFrame.set_index)


    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    return df.set_index(*args, **kwargs)


# split_dataframe() {{{1
def split_dataframe(df: pd.DataFrame,
                    chunk_size: int = 1000) -> List:
    ''' Split dataframe by chunk_size rows, returning multiple dataframes

    1. Define 'range' (start, stop, step/chunksize)
    2. Use np.split() to examine dataframe indices using the calculated 'range'.

    Parameters
    ----------
    df
        dataframe to be split
    chunksize
        default=1000

    Returns
    -------
    A list of pd.DataFrame 'chunks'

    Examples
    --------

    .. code-block::

        chunks = split(customer_orders_tofix, 1000)
        for df in chunks:
            display(head(df, 2))

    '''
    nrows = df.shape[0]
    range_ = range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)
    logger.debug(range_)

    return np.split(df, range_)


# str_join() {{{1
def str_join(df: pd.DataFrame,
                columns: List = None,
                column: str = None,
                sep: str = '',
                loc: str = 'after',
                drop: bool = True) -> pd.DataFrame:
    ''' join or combine columns with a separator

    Join or combine a number of columns together into one column. Column(s) that
    are not of 'string' data type are automatically converted to strings then
    combined.

    .. note::

        If the column keyword value contains one of the combined columns, it will
        automatically replace the original column if drop=True.

        If drop=False, the function will rename and append an underscore to the
        returned combined column name. See example for details:

    Parameters
    ----------
    df
        a pandas dataframe
    columns
        list of column names to join/combine
    column
        (optional) column name to store the results of combined columns
    sep
        (optional) separator value. Default is ''.
    loc
        location to place the split column output within the dataframe
        Default is 'after' meaning place the output after the column.

        Valid locations are: 'before' or 'after' the column.
        You can also specify 'first' or 'last' corresponding to the first and last
        columns of the dataframe.

        For more information about relocating columns - see the relocate() function.
    drop
        drop original columns used in str_join function

    Returns
    -------
    A copy of the pandas dataframe

    Examples
    --------

    .. code-block::

        %%piper
        sample_sales()
        >> str_join(columns=['actual_sales', 'product', 'actual_profit'],
                    sep='|',
                    column='actual_sales',
                    drop=True)
        >> head(tablefmt='plain')

             location month               target_sales target_profit actual_sales
          4  London   2021-01-01 00:00:00        31749          1905 29209.08|Beachwear|1752.54
        125  London   2021-01-01 00:00:00        37833          6053 34049.7|Beachwear|5447.95
         21  London   2021-01-01 00:00:00        29485          4128 31548.95|Jeans|4416.85
        148  London   2021-01-01 00:00:00        37524          3752 40901.16|Jeans|4090.12

    Same example, this time with drop=False, note the appended underscore in the
    combined column name.

    .. code-block::

        %%piper

        sample_sales()
        >> select(['-target_profit', '-month'])
        >> str_join(columns=['actual_sales', 'product', 'actual_profit'],
                    sep='|',
                    column='actual_sales',
                    drop=False)
        >> head(tablefmt='plain')

             location product   target_sales actual_sales actual_sales_              actual_profit
          4  London   Beachwear        31749        29209 29209.08|Beachwear|1752.54          1753
        125  London   Beachwear        37833        34050 34049.7|Beachwear|5447.95           5448
         21  London   Jeans            29485        31549 31548.95|Jeans|4416.85              4417
        148  London   Jeans            37524        40901 40901.16|Jeans|4090.12              4090

    '''
    df = _dataframe_copy(df, inplace=False)

    if not isinstance(columns, list):
            raise NameError(f"Columns '{columns}' must be a list")

    if column is not None:
        if not isinstance(column, str):
            raise TypeError(f"Column name '{column}' must be a string")

    if len(columns) < 2:
        raise ValueError(f"Please enter at least 2 columns")

    for col in columns:
        if col not in df.columns.tolist():
                raise NameError(f"Please check column name '{col}' specified")

    # Make sure ALL columns to be concatenated are string type
    for col in columns:
        if not pd.api.types.is_string_dtype(df[col]) and \
               not pd.api.types.is_object_dtype(df[col]):

            df[col] = df[col].astype(str)

    new_col = df[columns[0]].str.cat(df[columns[1]], sep=sep)

    if column is None:
        new_col.name = '0'
    else:
        new_col.name = column

    # If one of the columns to be combined has the same name
    # as the new column, append '_' to the combined column name.
    duplicate_column_name = False
    for idx, col in enumerate(columns):
        if col in new_col.name:
            new_col.name = new_col.name + '_'
            duplicate_column_name = True

    if len(columns) > 2:
        for col in columns[2:]:
            new_col = new_col.str.cat(df[col], sep=sep)

    df = pd.concat([df, new_col], axis=1)

    df = relocate(df, new_col.name, loc=loc, ref_column=columns[0])

    if drop:
        df = df.drop(columns=columns)

        if duplicate_column_name:
            df = df.rename(columns={new_col.name: column})

    return df


# str_split() {{{1
def str_split(df: pd.DataFrame,
              column: str = None,
              columns: List = None,
              pat: str = ',',
              n: int = -1,
              expand: bool = True,
              loc: str = 'after',
              drop: bool = False) -> pd.DataFrame:
    ''' split column

    Function accepts a column to split, a pattern/delimitter value and optional
    list of column names to store the result. By default the result is placed just
    after the specified column.

    .. note::

        If one of the target split columns contains the same name as the column
        to be split and drop=False, the function will append an underscore '_'
        to the end of the corresponding new split column name.

        If drop=True, the the new split column name will NOT be renamed and just
        replace the original column name. See examples for details.

    Parameters
    ----------
    df
        a pandas dataframe
    column
        column to be split
    columns
        list-like of column names to store the results of the split
        column values
    pat
        regular expression pattern. Default is ','

        .. note::

            For space(s), safer to use r'\\s' rather than ' '.

    n
        default -1. Number of splits to capture. -1 means capture ALL splits
    loc
        location to place the split column output within the dataframe
        Default is 'after' meaning place the output after the column.

        Valid locations are: 'before' or 'after' the column.
        You can also specify 'first' or 'last' corresponding to the first and last
        columns of the dataframe.

        For more information about relocating columns - see the relocate() function.
    drop
        drop original column to be split

    Returns
    -------
    A copy of the pandas dataframe

    Examples
    --------
    An example where the one of the new split column names is the same as the
    split column.

    .. code-block::

        %%piper

        sample_sales()
        >> select(['-target_profit', '-actual_sales'])
        >> str_split(column='month',
                    pat='-',
                    columns=['day', 'month', 'year'],
                    drop=False)
        >> head(tablefmt='plain')

             location product    month        day month_ year target_sales actual_profit
          4  London   Beachwear  2021-01-01  2021     01   01        31749          1753
        125  London   Beachwear  2021-01-01  2021     01   01        37833          5448
         21  London   Jeans      2021-01-01  2021     01   01        29485          4417
        148  London   Jeans      2021-01-01  2021     01   01        37524          4090

    The same example, this time with the drop=True parameter specified.

    .. code-block::

        sample_sales()
        >> select(['-target_profit', '-actual_sales'])
        >> str_split(column='month',
                    pat='-',
                    columns=['day', 'month', 'year'],
                    drop=True)
        >> head(tablefmt='plain')

             location    product      day    month    year    target_sales    actual_profit
          4  London      Beachwear   2021       01      01           31749             1753
        125  London      Beachwear   2021       01      01           37833             5448
         21  London      Jeans       2021       01      01           29485             4417
        148  London      Jeans       2021       01      01           37524             4090

    '''
    df = _dataframe_copy(df, inplace=False)

    if not isinstance(column, str):
        raise TypeError(f"Column name '{column}' must be a string")

    if column not in df.columns:
        raise NameError(f"Please check column name '{column}' specified")

    if columns is not None:
        if not isinstance(columns, list):
            raise TypeError(f"Columns '{columns}' must be list-like")

    # Make sure column to be split is of string type
    if not pd.api.types.is_string_dtype(df[column]) and \
           not pd.api.types.is_object_dtype(df[column]):
        df[column] = df[column].astype(str)

    if not expand:
        df[column] = df[column].str.split(pat=pat, n=n, expand=False)
    else:
        split_cols = df[column].str.split(pat=pat, n=n, expand=True)

        duplicate_column_name = False

        if columns is None:
            df = pd.concat([df, split_cols], axis=1)
            cols_to_move = split_cols.columns.tolist()
            df = relocate(df, cols_to_move, loc=loc, ref_column=column)
        else:
            # If one of the new split columns has the same name
            # as the original column, append '_' to the split column name.
            for idx, col in enumerate(columns):
                if col in column:
                    columns[idx] = columns[idx] + '_'
                    duplicate_column_name = True

            split_cols.columns = columns
            df = pd.concat([df, split_cols], axis=1)
            df = relocate(df, columns, loc=loc, ref_column=column)

        if drop:
            df = df.drop(columns=column)

            if duplicate_column_name:
                df = df.rename(columns={column + '_': column})

    return df


# summarise() {{{1
def summarise(df: pd.DataFrame,
              *args,
              **kwargs) -> pd.DataFrame:
    '''summarise or aggregate data.

    This is a wrapper function rather than using e.g. df.agg()
    For details of args, kwargs - see help(pd.DataFrame.agg)


    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame


    Notes
    -----
    Different behaviour to pd.DataFrame.agg()

    If only the dataframe is passed, the function will
    do a 'count' and 'sum' of all columns. See also info().

    .. code-block::

        %piper sample_sales() >> summarise()

          names             n          sum
          location        200          nan
          product         200          nan
          month           200          nan
          target_sales    200   5423715.00
          target_profit   200    487476.42
          actual_sales    200   5376206.33
          actual_profit   200    482133.92

    If you pass a groupby object to summarise() with no other
    parameters - summarise will summate all numeric columns.

    The equivalent to:
    %pipe df >> group_by(['col1', 'col2']) >> summarise(sum)

    .. code-block::

        %%piper
        sample_sales()
        >> group_by(['location', 'product'])
        >> summarise()
        >> head(tablefmt='plain')

                                    target_sales    target_profit    actual_sales    actual_profit
        ('London', 'Beachwear')           407640            41372          388762            38940
        ('London', 'Footwear')            275605            21449          274674            21411
        ('London', 'Jeans')               414488            42200          404440            40691
        ('London', 'Sportswear')          293384            28149          291561            28434


    Examples
    --------
    Syntax 1

    .. code-block::

        column_name = ('existing_column', function)

    .. note::

        below there are some 'common' functions that can be quoted like 'sum',
        'mean', 'count', 'nunique' or just state the function name

    .. code-block::

        %%piper
        sample_sales() >>
        group_by('product') >>
        summarise(totval1=('target_sales', sum),
                  totval2=('actual_sales', 'sum'))


    Syntax 2

    .. code-block::

        column_name = pd.NamedAgg('existing_column', function)

    .. code-block::

        %%piper
        sample_sales() >>
        group_by('product') >>
        summarise(totval1=(pd.NamedAgg('target_sales', 'sum')),
                  totval2=(pd.NamedAgg('actual_sales', 'sum')))


    Syntax 3

    .. code-block::

        {'existing_column': function}
        {'existing_column': [function1, function2]}

    .. code-block::

        %%piper
        sample_sales()
        >> group_by('product')
        >> summarise({'target_sales':['sum', 'mean']})

    Syntax 4:

    .. code-block::

        'existing_column': lambda x: x+1

    Example below identifies unique products sold by location.

    .. code-block::

        %%piper
        sample_sales() >>
        group_by('location') >>
        summarise({'product': lambda x: set(x.tolist())}) >>

        # Alternative coding of 'list' ;)
        # summarise(products=('product', lambda x: list(set(x)))) >>

        # explode('product')

    '''
    # If nothing passed to summarise - return a count and sum of all columns
    if not kwargs and len(args) == 0:

        if isinstance(df, pd.DataFrame):
            logger.info("Use group_by() and summarise() for more detail")
            sum_ = df.select_dtypes(include='number').agg('sum')
            count_ = df.agg('count')
            summary_ = pd.concat([count_, sum_], axis=1)
            summary_.columns = ['n', 'sum']
            summary_.index.names = ['names']
            return summary_

        # If groupby obj, assume sum function
        if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy ):
                return df.agg(sum)

    group_df = df.agg(*args, **kwargs)

    return group_df


# summary_df() {{{1
def summary_df(datasets,
               title = 'Summary',
               col_total = 'Total records',
               add_grand_total = True,
               grand_total = 'Grand total'):
    '''Summarise a dictionary of dataframes.

    Summarises a list of dataframes and (optionally) provide a grand total row
    sum.

    Parameters
    ----------
    datasets
        a list tuples containing dataframe description and dataframe(s)
    title
        title to be used in the summary dataframe.
    col_total
        column total title
    add_grand_total
        Default True
    grand_total
        grand total title


    Returns
    -------
    pandas dataframe containing summary info


    Examples
    --------

    .. code-block::

        datasets = [('1st dataset', df), ('2nd dataset', df2)]
        summary_df = summary_df(datasets, title='Summary',
                                col_total='Total records',
                                add_grand_total=True,
                                grand_total='Grand total')
    '''
    today_str = f"@ {datetime.now().strftime('%d, %b %Y')}"
    summary_title = ' '.join((title, today_str))

    dict1 = {descr: dataset.shape[0] for descr, dataset in datasets}
    summary = pd.DataFrame(data=dict1, index=range(1)).T
    summary.columns = [col_total]
    summary.rename_axis(summary_title, axis=1, inplace=True)

    if add_grand_total:
        s1 = pd.Series({col_total: summary[col_total].sum()},
                       name=grand_total)
        summary = summary.append(s1)

    return summary


# stack() {{{1
def stack(df: pd.DataFrame,
          *args,
          **kwargs) -> pd.DataFrame:
    '''stack dataframe

    This is a wrapper function rather than using e.g. df.stack()
    For details of args, kwargs - see help(pd.DataFrame.stack)


    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    return df.stack(*args, **kwargs)


# tail() {{{1
def tail(df: pd.DataFrame,
         n: int = 4,
         shape: bool = True,
         tablefmt: bool = False,
         precision: int = 0) -> pd.DataFrame:
    '''show last n records of a dataframe

    Like the corresponding R function, displays the last n records
    Alternative to df.tail().

    Examples
    --------
    .. code-block::

        tail(df)


    Parameters
    ----------
    df
        pandas dataframe
    n
        Default n=4. number of rows to display
    shape
        Default True, show shape information
    tablefmt
        Default False. If supplied, tablefmt keyword value can be any valid
        format supplied by the tabulate package
    precision
        Default 0. Number precision shown if tablefmt specified

    Returns
    -------
    A pandas dataframe
    '''
    if shape:
        _shape(df)

    if tablefmt:
        print(df.tail(n=n).to_markdown(tablefmt=tablefmt, floatfmt=f".{precision}f"))
        return

    return df.tail(n=n)


# transform() {{{1
def transform(df: pd.DataFrame,
              index: Union[str, List[str]] = None,
              **kwargs) -> pd.DataFrame:
    '''Add a group calculation to grouped DataFrame

    Transform is based on the pandas pd.DataFrame.transform() function.
    For details of args, kwargs - see help(pd.DataFrame.transform)

    Based on the given dataframe and grouping index, creates new aggregated column
    values using the list of keyword/value arguments (kwargs) supplied.

    By default, it calculates the group % of the first numeric column, in
    proportion to the first 'index' or grouping value.

    Examples
    --------
    Calculate group percentage value

    .. code-block::

        %%piper
        sample_data() >>
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

    .. code-block::

        %%piper
        sample_data() >>
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
    df
        dataframe to calculate grouped value
    index
        grouped column(s) (str or list) to be applied as grouping index.
    kwargs
        Similar to 'assign', keyword arguments to be assigned as dataframe
        columns containing, tuples of column_name and function e.g.
        new_column=('existing_col', 'sum')

        If no kwargs supplied - calculates the group percentage ('g%') using
        the first index column as index key and the first column value(s).


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


# str_trim() {{{1
def str_trim(df: pd.DataFrame, str_columns: list = None) -> pd.DataFrame:
    '''strip leading/trailing blanks

    Parameters
    ----------
    df
        pandas dataframe
    str_columns
        Optional list of columns to strip. If None, all string data type columns
        will be trimmed.

    Returns
    -------
    A pandas dataframe
    '''
    if str_columns is None:
        str_columns = df.select_dtypes(include='object')

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
        df[col] = df[col].astype('str').str.strip()

    return df


# unstack() {{{1
def unstack(df: pd.DataFrame,
          *args,
          **kwargs) -> pd.DataFrame:
    '''unstack dataframe

    This is a wrapper function rather than using e.g. df.unstack()
    For details of args, kwargs - see help(pd.DataFrame.unstack)


    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    return df.unstack(*args, **kwargs)


# where() {{{1
def where(df: pd.DataFrame,
          *args,
          **kwargs) -> pd.DataFrame:
    '''where/filter dataframe rows

    This is a wrapper function rather than using e.g. df.query()
    For details of args, kwargs - see help(pd.DataFrame.query)

    Examples
    --------
    .. code-block::

        (select(customers)
         .pipe(clean_columns)
         .pipe(select, ['client_code', 'establishment_type', 'address_1', 'address_2', 'town'])
         .pipe(where, 'establishment_type != 91')
         .pipe(where, "town != 'LISBOA' & establishment_type != 91"))

    .. code-block::

        %%piper
        get_sample_data()
        >> str_trim()
        >> select('-dates')
        >> where(""" ~countries.isin(['Italy', 'Portugal']) &
                    values_1 > 40 &
                    values_2 < 25 """)
        >> order_by('-countries')
        >> head(5)

    Parameters
    ----------
    df
        dataframe
    *args
        arguments for wrapped function
    **kwargs
        keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    logger.debug(args)
    logger.debug(kwargs)

    args_copy = list(args)
    args_copy[0] = re.sub('\n', '', args_copy[0])

    try:
        return df.query(*args_copy, **kwargs)
    except (pd.core.computation.ops.UndefinedVariableError) as e:
        logger.info("ERROR: External @variable? use: where(..., global_dict=globals())")

# _check_col() {{{1
def _check_col(column: str,
               selected: List,
               drop: List,
               columns: List) -> Tuple[List, List]:
    ''' Check column add/remove from column list

    Used by select() function.

    Parameter
    ---------
    column
        column to be added/removed
    selected
        list of columns to be selected
    drop
        list of columns to be dropped
    columns
        all columns within dataframe (for validation)

    Returns
    -------
    list
        updated select column list
    '''
    remove = False

    if isinstance(column, str):
        if column.startswith('-'):
            remove = True
            column = column[1:]

    if column not in columns:
        raise KeyError(f"Column '{column}' not found!")

    if remove:
        if column not in drop:
            drop.append(column)
    else:
        if column not in selected:
            selected.append(column)

    return selected, drop


# _set_grouper() {{{1
def _set_grouper(df: pd.DataFrame,
                index: Union[Any, List[Any]],
                freq: str = 'D') -> pd.DataFrame:
    '''Convert index to grouper index

    For given dataframe and index (str, list), return a dataframe with
    a pd.Grouper object for each defined index.


    Parameters
    ----------
    df
        dataframe
    index
        index column(s) to be converted to pd.Grouper objects
    freq
        Default 'd' (days)

        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        for list of valid frequency strings.


    Returns
    -------
    A pandas DataFrame
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


# _dataframe_copy() {{{1
def _dataframe_copy(df: pd.DataFrame,
                    inplace: int = False) -> pd.DataFrame:
    '''Return a copy of the dataframe

    Helper function to return either a copy of dataframe or a reference to the
    passed dataframe based on the passed 'inplace' parameter.

    Basically, if we need to implement 'inplace' keyword in a main function, we
    can just copy the below line, and the function will either work directly on
    the received dataframe or a copy.

    df = _dataframe_copy(df, inplace=False)

    Parameters
    ----------
    df
        pandas dataframe

    Returns
    -------
    A pandas dataframe
    '''
    if inplace:
        return df

    return df.copy(deep=True)


# _shape() {{{1
def _shape(df: pd.DataFrame) -> pd.DataFrame:
    '''Show shape information

    Parameters
    ----------
    df
        pandas dataframe


    Returns
    -------
    A pandas dataframe
    '''
    if isinstance(df, pd.Series):
        logger.info(f'{df.shape[0]} rows')

    if isinstance(df, pd.DataFrame):
        logger.info(f'{df.shape[0]} rows, {df.shape[1]} columns')
