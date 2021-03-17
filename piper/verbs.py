from datetime import datetime
from pandas.api.types import is_datetime64_any_dtype
from pandas.api.types import is_period_dtype
from pandas.core.common import flatten
from piper.decorators import shape
from piper.io import _file_with_ext
from piper.xl import WorkBook
import logging
import numpy as np
import pandas as pd
import re
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


logger = logging.getLogger(__name__)


# adorn() {{{1
def adorn(df: pd.DataFrame,
          columns: Union[str, list] = None,
          fillna: Union[str, int] = '',
          col_row_name: str = 'All',
          axis: Union [int, str] = 0,
          ignore_row_index: bool = False) -> pd.DataFrame:
    ''' <*> add totals to a dataframe
    Based on R janitor package function add row and/or column totals to a
    dataframe.


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

    Usage example
    -------------
    ```python
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
    ```

    |                     |   East |   North |   South |   West |    All |
    |:--------------------|-------:|--------:|--------:|-------:|-------:|
    | Children's Clothing |  45849 |   37306 |   18570 |  20182 | 121907 |
    | Men's Clothing      |  51685 |   39975 |   18542 |  19077 | 129279 |
    | Women's Clothing    |  70229 |   61419 |   22203 |  22217 | 176068 |
    | All                 | 167763 |  138700 |   59315 |  61476 | 427254 |

    '''
    # ROW:
    if axis == 0 or axis == 'row' or axis == 'both':

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

            # If single index, return that else return a tuple of the
            # combined multi-index(key)
            index_length = len(df.index.names)
            if index_length == 1:
                total_row = [col_row_name]
            else:
                total_row = ['' for _ in df.index.names]
                total_row[index_length - 1] = col_row_name

            totals = pd.DataFrame(totals, index=total_row)
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

# add_xl_formula() {{{1
def add_xl_formula(df: pd.DataFrame,
                   column_name: str = 'xl_calc',
                   formula: str = '=CONCATENATE(A{row}, B{row}, C{row})',
                   offset: int = 2) -> pd.DataFrame:

    ''' <*> add Excel (xl) formula column

    Parameters
    ----------
    df : pandas dataframe

    column_name : the column name to be associated with the column formula
                  values, default 'xl_calc'

    formula : Excel formula to be applied

              e.g. '=CONCATENATE(A{row}, B{row}, C{row})'
              {row} is the defined replacement variable which will be replaced
              with actual individual row value.

    offset : starting row value, default = 2 (resultant xl sheet includes headers)


    Usage example
    -------------
    ```python
    formula = '=CONCATENATE(A{row}, B{row}, C{row})'
    add_xl_formula(df, column_name='X7', formula=formula)
    ```


    Returns
    -------
    pandas dataframe
    '''

    col_values = []
    for x in range(offset, df.shape[0] + offset):
        repl_str = re.sub('{ROW}', str(x), string=formula, flags=re.I)
        col_values.append(repl_str)

    df[column_name] = col_values

    return df


# across {{{1
def across(df: pd.DataFrame,
               columns: Union[str, List[str]],
               function: Callable,
               apply_values: bool = True,
               *args, **kwargs) -> pd.DataFrame:
    ''' Apply a function across multiple columns

    Usage example
    -------------
    %%piper
    sample_data()
    >> across(['dates', 'order_dates'], to_julian)
    >> head()

    |   dates |   order_dates | countries   | regions   | ids   |   values_1 |   values_2 |
    |--------:|--------------:|:------------|:----------|:------|-----------:|-----------:|
    |  120001 |        120007 | Italy       | East      | A     |        311 |         26 |
    |  120002 |        120008 | Portugal    | South     | D     |        150 |        375 |
    |  120003 |        120009 | Spain       | East      | A     |        396 |         88 |
    |  120004 |        120010 | Italy       | East      | B     |        319 |        233 |


    %%piper
    sample_data()
    >> across(['dates', 'order_dates'], fiscal_year, year_only=True)
    >> head()

    | dates    | order_dates   | countries   | regions   | ids   |   values_1 |   values_2 |
    |:---------|:--------------|:------------|:----------|:------|-----------:|-----------:|
    | FY 19/20 | FY 19/20      | Italy       | East      | A     |        311 |         26 |
    | FY 19/20 | FY 19/20      | Portugal    | South     | D     |        150 |        375 |
    | FY 19/20 | FY 19/20      | Spain       | East      | A     |        396 |         88 |
    | FY 19/20 | FY 19/20      | Italy       | East      | B     |        319 |        233 |


    Parameters
    ----------
    df : pandas dataframe

    columns : column(s) to apply function

    function : function to be called.

    apply_values : Default True.
        True - function assumes that it should be applied to each Series row values.
        False - function assumes called function applied at Series 'object' level.


    Return
    ------
    A pandas dataframe
    '''
    if columns is None:
        raise ValueError('Please specify function to apply')

    if isinstance(df, pd.Series):
        raise TypeError('Please specify DataFrame object')

    if function is None:
        raise ValueError('Please specify function to apply')

    if isinstance(columns, str):
        if columns not in df.columns:
            raise ValueError(f'column {columns} not found')

    if isinstance(columns, list):
        for col in columns:
            if col not in df.columns:
                raise ValueError(f'column {col} not found')

    if isinstance(columns, str):
        df[columns] = df[columns].apply(function, *args, **kwargs)

    # For multiple columns, does the user want to:

    # 1. Apply the function to the Series values
    if apply_values:
        if isinstance(columns, list):
            for col in columns:
                df[col] = df[col].apply(function, *args, **kwargs)

    # 2. Access the Series vectorized functions (e.g. str, astype etc.)
    else:
        df[columns] = df[columns].apply(function, *args, **kwargs)

    return df


# assign() {{{1
def assign(df: pd.DataFrame,
           *args,
           str_to_lambdas: bool = True,
           lambda_var: str = 'x',
           info: bool = False,
           **kwargs) -> pd.DataFrame:
    ''' <*> Assign new columns to a DataFrame.

    Returns a new object with all original columns in addition to new ones.
    Existing columns that are re-assigned will be overwritten.

    Parameters
    ----------
    str_to_lambdas: boolean - default is False.

    lambda_var: str - default is 'x'.

    info: boolean - default is False.
        if True - output contents of kwargs where keyword value is being
        converted from string to lambda function. (For debug purposes)

    **kwargs : dict of {str: callable or Series}
        The column names are keywords. If the values are
        callable, they are computed on the DataFrame and
        assigned to the new columns. The callable must not
        change input DataFrame (though pandas doesn't check it).
        If the values are not callable, (e.g. a Series, scalar, or array),
        they are simply assigned.


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


    Usage example
    -------------
    You can create a dictionary of column names to corresponding functions,
    then pass the dictionary to the assign function as shown below:

    ```python
    %%piper --dot

    sample_data()
    .. assign(**{'reversed': 'x.regions.apply(lambda x: x[::-1])',
                 'v1_x_10': 'x.values_1 * 10',
                 'v2_div_4': lambda x: x.values_2 / 4,
                 'dow': lambda x: x.dates.dt.day_name(),
                 'ref': lambda x: x.v2_div_4 * 5,
                 'values_1': 'x.values_1.astype(int)',
                 'values_2': 'x.values_2.astype(int)',
                 'ids': lambda x: x.ids.astype('category')})
    .. pd.DataFrame.astype({'values_1': float, 'values_2': float})
    .. relocate('dow', 'after', 'dates')
    .. select(['-dates', '-order_dates'])
    .. head()
    ```

    | dow      | countries| regions| ids|values_1 |values_2 | reversed|v1_x_10 |v2_div_4 |    ref |
    |:---------|:---------|:-------|:---|--------:|--------:|:--------|-------:|--------:|-------:|
    | Wednesday| Italy    | East   | A  |     311 |      26 | tsaE    |   3110 |    6.5  |  32.5  |
    | Thursday | Portugal | South  | D  |     150 |     375 | htuoS   |   1500 |   93.75 | 468.75 |
    | Friday   | Spain    | East   | A  |     396 |      88 | tsaE    |   3960 |   22    | 110    |
    | Saturday | Italy    | East   | B  |     319 |     233 | tsaE    |   3190 |   58.25 | 291.25 |

    %%piper
    sample_sales()
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

    df = df.assign(*args, **kwargs)

    return df


# columns() {{{1
def columns(df: pd.DataFrame,
            regex: str = None,
            astype: str = 'list') -> Union[str, list, dict, pd.Series, pd.DataFrame]:
    ''' <*> show dataframe column information

    This function is useful reviewing or manipulating column(s)

    Usage example
    -------------
    ```python
    import numpy as np
    import pandas as pd
    from pandas._testing import assert_frame_equal

    id_list = ['A', 'B', 'C', 'D', 'E']
    s1 = pd.Series(np.random.choice(id_list, size=5), name='ids')

    region_list = ['East', 'West', 'North', 'South']
    s2 = pd.Series(np.random.choice(region_list, size=5), name='regions')

    df = pd.concat([s1, s2], axis=1)

    expected = ['ids', 'regions']
    actual = columns(df, astype='list')
    assert expected == actual

    expected = {'ids': 'ids', 'regions': 'regions'}
    actual = columns(df, astype='dict')
    assert expected == actual

    expected = pd.DataFrame(['ids', 'regions'], columns=['column_names'])
    actual = columns(df, astype='dataframe')
    assert_frame_equal(expected, actual)

    expected = "['ids', 'regions']"
    actual = columns(df, astype='text')
    assert expected == actual
    ```

    Parameters
    ----------
    df : dataframe

    regex : regular expression to 'filter' list of returned columns
            default, None

    astype : default - list. See return options below


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
    ''' <*> show column/category/factor frequency

    For selected column or multi-index show the frequency count, frequency %,
    cum frequency %. Also provides optional totals.

    Usage example
    -------------
    ```python
    import numpy as np
    import pandas as pd
    from piper.verbs import count

    np.random.seed(42)

    id_list = ['A', 'B', 'C', 'D', 'E']
    s1 = pd.Series(np.random.choice(id_list, size=5), name='ids')
    s2 = pd.Series(np.random.randint(1, 10, s1.shape[0]), name='values')
    df = pd.concat([s1, s2], axis=1)

    count(df.ids)
    ```

    | ids   |   n |   % |   cum % |
    |:------|----:|----:|--------:|
    | E     |   3 |  60 |      60 |
    | C     |   1 |  20 |      80 |
    | D     |   1 |  20 |     100 |


    Parameters
    ----------
    df : dataframe reference

    columns: dataframe columns/index to be used in groupby function

    totals_name: name of total column, default 'n'

    percent: provide % total, default True

    cum_percent: provide cum % total, default True

    threshold: filter cum_percent by this value, default 100

    round = round decimals, default 2

    totals: add total column, default False

    sort_values: default False, None means use index sort

    reset_index: default False


    Returns
    -------
    A pandas dataframe
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


# clean_columns() {{{1
def clean_columns(df: pd.DataFrame,
                  replace_char: tuple = (' ', '_'),
                  title: bool = False) -> pd.DataFrame:
    ''' <*> Clean column names, strip blanks, lowercase, snake_case.

    Also removes awkward characters to allow for easier manipulation.
    (optionally 'title' each column name.)

    Usage example
    -------------
    ```python
    df = clean_columns(df, replace_char=('_', ' '))
    ```


    Parameters
    ----------
    df : dataframe

    replace_char : a tuple giving the 'from' and 'to' characters
                   to be replaced, default (' ', '_')

    title : default False. If True, titleize column values


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
    ''' <*> Combine 1 or more header rows across into one.

    Optionally, infers remaining data column data types.

    Usage example
    -------------

    | A     | B      |
    |:------|:-------|
    | Order | Order  |
    | Qty   | Number |
    | 10    | 12345  |
    | 40    | 12346  |

    ```python
    data = {'A': ['Order', 'Qty', 10, 40],
            'B': ['Order', 'Number', 12345, 12346]}
    df = pd.DataFrame(data)
    df = combine_header_rows(df)
    df
    ```

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

    title : default False. If True, titleize column values

    infer_objects : default True. Infer data type of resultant dataframe


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
    ''' <*> select distinct/unique rows

    This is a wrapper function rather than using e.g. df.distinct()
    For details of args, kwargs - see help(pd.DataFrame.distinct)

    Usage example
    -------------
    ```python
    from piper.verbs import select, distinct
    from piper.test.factory import get_sample_df1

    df = get_sample_df1()
    df = select(df, ['countries', 'regions', 'ids'])
    df = distinct(df, 'ids', shape=False)

    expected = (5, 3)
    actual = df.shape
    ```


    Parameters
    ----------
    df: dataframe

    shape : default True
        show shape information as a logger.info() message

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


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
    ''' <*> drop column(s)

    This is a wrapper function rather than using e.g. df.drop()
    For details of args, kwargs - see help(pd.DataFrame.drop)

    Usage example
    -------------
    ```python
    df <- pd.read_csv('inputs/export.dsv', sep='\t')
    >> clean_columns()
    >> trim()
    >> assign(adast = lambda x: x.adast.astype('category'),
              adcrcd = lambda x: x.adcrcd.astype('category'),
              aduom = lambda x: x.aduom.astype('category'))
    >> drop(columns='adcrcd_1')
    ```


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    return df.drop(*args, **kwargs)


# duplicated() {{{1
def duplicated(df: pd.DataFrame,
               subset: Union[str, List[str]] = None,
               keep: bool = False,
               sort: bool = True,
               column: str = 'duplicate',
               loc: str = 'first',
               ref_column: str = None,
               unique_only: bool = False) -> pd.DataFrame:
    ''' <*> locate duplicate data

    Usage example
    -------------
    ```python
    from piper.test.factory import get_sample_df5
    df = get_sample_df5()

    duplicated_fields = ['bgy56icnt', 'bgz56ccode']
    df = duplicated(df[duplicated_fields],
                    subset=duplicated_fields,
                    keep='first',
                    sort=True,
                    column='dupe',
                    loc='first')
    df
    ```
    |    | dupe   | bgy56icnt   |   bgz56ccode |
    |---:|:-------|:------------|-------------:|
    |  0 | False  | BE          |     46065502 |
    |  1 | True   | BE          |     46065502 |
    |  2 | True   | BE          |     46065502 |
    |  3 | False  | BE          |     46065798 |
    |  4 | False  | BE          |     46066013 |


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


# explode() {{{1
def explode(df: pd.DataFrame,
            *args,
            **kwargs) -> pd.DataFrame:
    ''' <*> Transform list-like column values to rows

    This is a wrapper function rather than using e.g. df.drop()
    For details of args, kwargs - see help(pd.DataFrame.drop)

    Usage example
    -------------
    ```python
    from piper.test.factory import get_sample_df1

    df = get_sample_df1()
    df = group_by(df, 'countries')
    df = summarise(df, ids=('ids', set))
    df.head()
    ```

    | countries   | ids                       |
    |:------------|:--------------------------|
    | Italy       | {'B', 'C', 'D', 'A', 'E'} |
    | Portugal    | {'B', 'C', 'A', 'D', 'E'} |
    | Spain       | {'B', 'C', 'D', 'A', 'E'} |
    | Switzerland | {'B', 'C', 'A', 'D', 'E'} |
    | Sweden      | {'B', 'C', 'A', 'D', 'E'} |

    ```python
    explode(df, 'ids').head(8)
    ```

    | countries   | ids   |
    |:------------|:------|
    | Italy       | B     |
    | Italy       | C     |
    | Italy       | D     |
    | Italy       | A     |
    | Italy       | E     |
    | Portugal    | B     |
    | Portugal    | C     |
    | Portugal    | A     |

    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''

    return df.explode(*args, **kwargs)


# flatten_cols() {{{1
def flatten_cols(df: pd.DataFrame,
                 join_char: str = '_',
                 remove_prefix=None) -> pd.DataFrame:
    ''' <*> Flatten multi-index column headings

    Usage example
    -------------
    ```python
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
    ```


    Parameters
    ----------
    df: pd.DataFrame - dataframe

    join_char: delimitter joining 'prefix' value(s) to be removed.

    remove_prefix: string(s) (delimitted by pipe '|')
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
    ''' <*> format dataframe datelike index

    Checks if dataframe index contains a date-like grouper object.
    If so, applies the 'frequency' string given.

    Usage example
    -------------


    Parameters
    ----------
    df: dataframe

    freq: Default 'd' (days)
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
    ''' <*> Group by dataframe

    This is a wrapper function rather than using e.g. df.groupby()
    For details of args, kwargs - see help(pd.DataFrame.groupby)

    The first argument or by keyword are checked in turn and converted
    to pd.Grouper objects. If any group fields are 'date' like, they
    can be represented as various 'frequencies' types.

    See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    for list of valid frequency strings.


    Usage example
    -------------
    ```python
    %%piper

    sample_data()
    >> where("ids == 'A'")
    >> where("values_1 > 300 & countries.isin(['Italy', 'Spain'])")
    >> group_by(['countries', 'regions'])
    >> summarise(total_values_1=pd.NamedAgg('values_1', 'sum'))
    ```


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    freq: Default 'd' (days)

    **kwargs: keyword-parameters for wrapped function


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
def head(df: pd.DataFrame, n: int = 4, shape: bool = True) -> pd.DataFrame:
    ''' <*> Show first n records
    Like the corresponding R function, displays the first n records.
    Alternative to df.head()

    Usage example
    -------------
    ```python
    head(df)
    ```


    Parameters
    ----------
    df : pandas dataframe

    n : number of rows to display, default n=4

    shape: True, show shape information


    Returns
    -------
    A pandas dataframe
    '''
    if shape:
        _shape(df)

    return df.head(n=n)


# info() {{{1
def info(df,
         n_dupes: bool = False,
         fillna: bool = False,
         memory_info: bool = True) -> pd.DataFrame:
    ''' <*> show dataframe meta data
    Provides a summary of the structure of the dataframe data.

    Usage example
    -------------
    ```python
    import numpy as np
    import pandas as pd
    from piper.verbs import info

    np.random.seed(42)

    id_list = ['A', 'B', 'C', 'D', 'E']
    s1 = pd.Series(np.random.choice(id_list, size=5), name='ids')
    s2 = pd.Series(np.random.randint(1, 10, s1.shape[0]), name='values')
    df = pd.concat([s1, s2], axis=1)

    |    | columns   | type   |   n |   isna |   isnull |   unique |
    |---:|:----------|:-------|----:|-------:|---------:|---------:|
    |  0 | ids       | object |   5 |      0 |        0 |        3 |
    |  1 | values    | int64  |   5 |      0 |        0 |        4 |
    ```


    Parameters
    ----------
    df: a pandas dataframe

    n_dupes: column showing the numbers of groups of duplicate records
             default: False

    fillna: filter out only columns which contain nulls/np.nan
            default: False

    memory: If True, show dataframe memory consumption in mb


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


# inner_join() {{{1
def inner_join(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    ''' <*> inner_join => df (All) | df2 (All) matching records only

    This is a wrapper function rather than using e.g. df.merge(how='inner')
    For details of args, kwargs - see help(pd.DataFrame.merge)

    Usage example
    -------------
    ```python
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
    ```

    |   OrderNo | Status   | Type_   | description    | description_2   |
    |----------:|:---------|:--------|:---------------|:----------------|
    |      1001 | A        | SO      | Standing Order | another one     |
    |      1003 | A        | SO      | Standing Order | another one     |
    |      1002 | C        | SA      | Sales Order    | Arbitrary desc  |


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    kwargs['how'] = 'inner'
    logger.debug(f"{kwargs}")

    return df.merge(*args, **kwargs)


# left_join() {{{1
def left_join(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    ''' <*> left_join => df (All) | df2 (All/na) df always returned

    This is a wrapper function rather than using e.g. df.merge(how='left')
    For details of args, kwargs - see help(pd.DataFrame.merge)

    Usage example
    -------------
    ```python
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
    ```

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
    ''' <*> show dataframe consumed memory (mb)

    Usage example
    -------------
    ```python
    from piper.factory import sample_sales
    from piper.verbs import memory
    import piper
    memory(sample_sales())
    ```
    >> Dataframe consumes 0.03 Mb


    Parameters
    ----------
    df: a pandas dataframe


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
    ''' <*> check for non-alphanumeric characters

    Parameters
    ----------
    df : pandas dataframe

    col_name : name of column to be checked


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
    ''' <*> order/sequence dataframe

    @__TODO__ Review mypy validation rules and update if required

    This is a wrapper function rather than using e.g. df.sort_values()
    For details of args, kwargs - see help(pd.DataFrame.sort_values)

    The first argument and/or keyword 'by' is checked to see:
    - for each specified column value:
      If it starts with a minus sign, assume ascending=False

    Usage example
    -------------
    ```python
    %%piper

    sample_data()
    >> group_by(['countries', 'regions'])
    >> summarise(totalval1=('values_1', 'sum'))
    >> group_calc(index='countries')
    >> order_by(['countries', '-group%'])
    >> head(8)
    ```

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
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


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
    ''' <*> outer_join => df (All/na) | df2 (All/na) All rows returned

    This is a wrapper function rather than using e.g. df.merge(how='outer')
    For details of args, kwargs - see help(pd.DataFrame.merge)

    Usage example
    -------------
    ```python
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
    ```python

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


# overlaps() {{{1
def overlaps(df: pd.DataFrame,
             unique_key: Union[str, List[str]] = None,
             start: str = 'effective',
             end: str = 'expiry',
             overlaps: str = 'overlaps') -> pd.DataFrame:
    ''' <*> Analyse dataframe rows with overlapping date periods

    Usage example
    -------------
    ```python
    data = {'prices': [100, 200, 300],
            'contract': ['A', 'B', 'A'],
            'effective': ['2020-01-01', '2020-03-03', '2020-05-30'],
            'expired': ['2020-12-31', '2021-04-30', '2022-04-01']}

    df = pd.DataFrame(data)
    overlaps(df, start='effective', end='expired', unique_key='contract')
    ```

    |    |   prices | contract   | effective   | expired    | overlaps   |
    |---:|---------:|:-----------|:------------|:-----------|:-----------|
    |  0 |      100 | A          | 2020-01-01  | 2020-12-31 | True       |
    |  1 |      200 | B          | 2020-03-03  | 2021-04-30 | False      |
    |  2 |      300 | A          | 2020-05-30  | 2022-04-01 | True       |


    Parameters
    ----------
    df : dataframe

    unique_key: (str, list)
        column(s) that uniquely identify rows

    start:
        column that defines start/effective date, default 'effective_date'

    end:
        column that defines end/expired date, default 'expiry_date'

    overlaps: str - default 'overlaps'
        name of overlapping column containing True/False values


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


# pivot_table() {{{1
def pivot_table(df: pd.DataFrame,
                *args,
                freq: str = 'M',
                format_date: bool = False,
                **kwargs) -> pd.DataFrame:
    ''' <*> create Excel like pivot table

    This is a wrapper function rather than using e.g. df.pivot_table()
    For details of args, kwargs - see help(pd.DataFrame.pivot_table)

    Usage example
    -------------
    ```python
    from piper.verbs import pivot_table
    from piper.test.factory import get_sample_df1
    import piper.defaults

    df = get_sample_df1()

    index=['dates', 'order_dates', 'regions', 'ids']
    pvt = pivot_table(df, index=index, freq='Q', format_date=True)
    pvt.head()
    ```

    |                                       |   values_1 |   values_2 |
    |:--------------------------------------|-----------:|-----------:|
    | ('Mar 2020', 'Mar 2020', 'East', 'A') |    227.875 |     184.25 |
    | ('Mar 2020', 'Mar 2020', 'East', 'B') |    203.2   |     168    |
    | ('Mar 2020', 'Mar 2020', 'East', 'C') |    126     |     367    |
    | ('Mar 2020', 'Mar 2020', 'East', 'D') |    125.667 |     259    |
    | ('Mar 2020', 'Mar 2020', 'East', 'E') |    194     |     219    |


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


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
    ''' <*> move column(s) in a dataframe
    Based on the corresponding R function - relocate

    Usage example
    -------------
    ```python
    %%piper
    sample_sales()
    >> pd.DataFrame.set_index(['location', 'product'])
    >> relocate(column='location', loc='after', ref_column='product', index=True)
    >> head()
    ```

    **NOTE**
    If you omit the keyword parameters, it probably 'reads' better ;)
    >> relocate(location', 'after', 'product', index=True)


    Parameters
    ----------
    df: dataframe

    column: column name(s) to be moved

    loc: default -'last'. The relative location to move the column(s) to.
        Valid values are: 'first', 'last', 'before', 'after'.

    ref_column: = Default - None. Reference column

    index: default is False (column). If True, then the column(s) being moved are
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
    ''' <*> rename dataframe col(s)
    This is a wrapper function rather than using e.g. df.rename()

    For details of args, kwargs - see help(pd.DataFrame.rename)

    Usage example
    -------------
    ```python
    %%piper
    sample_sales()
    >> rename(columns={'product': 'item'})
    ```


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


    Returns
    -------
    pandas DataFrame object
    '''
    return df.rename(*args, **kwargs)


# rename_axis() {{{1
def rename_axis(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame :
    ''' <*> rename dataframe axis
    This is a wrapper function rather than using e.g. df.rename_axis()

    For details of args, kwargs - see help(pd.DataFrame.rename_axis)

    Usage example
    -------------
    ```python
    %%piper

    sample_sales()
    >> pivot_table(index=['location', 'product'], values='target_sales')
    >> rename_axis(('AAA', 'BBB'), axis='rows')
    >> head()
    ```

    |                          |   target_sales |
    |    AAA          BBB      |                |
    |:-------------------------|---------------:|
    | ('London', 'Beachwear')  |        31379.8 |
    | ('London', 'Footwear')   |        27302.6 |
    | ('London', 'Jeans')      |        28959.8 |
    | ('London', 'Sportswear') |        29466.4 |


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


    Returns
    -------
    pandas DataFrame object

    '''
    return df.rename_axis(*args, **kwargs)


# right_join() {{{1
def right_join(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    ''' <*> right_join => df (All/na) | df2 (All) df2 always returned

    This is a wrapper function rather than using e.g. df.merge(how='right')
    For details of args, kwargs - see help(pd.DataFrame.merge)

    Usage example
    -------------
    ```python
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
    ```

    |   OrderNo | Status   | Type_   | description    | description_2   |
    |----------:|:---------|:--------|:---------------|:----------------|
    |      1002 | C        | SA      | Sales Order    | Arbitrary desc  |
    |      1001 | A        | SO      | Standing Order | another one     |
    |      1003 | A        | SO      | Standing Order | another one     |

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
    ''' <*> show sample data
    This is a wrapper function rather than using e.g. df.sample()

    For details of args, kwargs - see help(pd.DataFrame.sample)

    Usage example
    -------------
    ```python
    sample(df)
    ```


    Parameters
    ----------
    df : dataframe

    shape : show shape information as a logger.info() message

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


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
           expr: str = None,
           regex: bool = False) -> pd.DataFrame:
    ''' <*> select dataframe columns
    Based on select() function from R tidyverse.

    Given dataframe, select column names

    Usage examples
    --------------
    ```python
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
    ```


    Parameters
    ----------
    df: dataframe

    expr : default None
           str - single column (in quotes)
           list -> list of column names (in quotes)

           NOTE: prefixing column name with a minus sign
                 filters out the column from returned list of columns

    regex : default False. If True, treat column string as a regex

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


# set_columns() {{{1
def set_columns(df: pd.DataFrame,
                columns: Union[str, Any] = None) -> pd.DataFrame:
    ''' <*> set dataframe column names

    Parameters
    ----------
    df: dataframe

    columns: column(s) values to be changed in referenced dataframe


    Returns
    -------
    A pandas dataframe
    '''
    df.columns = columns

    return df


# where() {{{1
def where(df: pd.DataFrame,
          *args,
          **kwargs) -> pd.DataFrame:
    ''' <*> where/filter dataframe rows

    This is a wrapper function rather than using e.g. df.query()
    For details of args, kwargs - see help(pd.DataFrame.query)

    Usage example
    -------------
    ```python
    (select(customers)
     .pipe(clean_columns)
     .pipe(select, ['client_code', 'establishment_type', 'address_1', 'address_2', 'town'])
     .pipe(where, 'establishment_type != 91')
     .pipe(where, "town != 'LISBOA' & establishment_type != 91"))
    ```


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


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


# reset_index() {{{1
def reset_index(df: pd.DataFrame,
          *args,
          **kwargs) -> pd.DataFrame:
    ''' <*> reset_index dataframe

    This is a wrapper function rather than using e.g. df.reset_index()
    For details of args, kwargs - see help(pd.DataFrame.reset_index)


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    return df.reset_index(*args, **kwargs)


# set_index() {{{1
def set_index(df: pd.DataFrame,
          *args,
          **kwargs) -> pd.DataFrame:
    ''' <*> set_index dataframe

    This is a wrapper function rather than using e.g. df.set_index()
    For details of args, kwargs - see help(pd.DataFrame.set_index)


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    return df.set_index(*args, **kwargs)


# summarise() {{{1
def summarise(df: pd.DataFrame,
              *args,
              **kwargs) -> pd.DataFrame:
    ''' <*> summarise or aggregate data.

    This is a wrapper function rather than using e.g. df.agg()
    For details of args, kwargs - see help(pd.DataFrame.agg)


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame


    Notes
    -----
    Different behaviour to pd.DataFrame.agg()

    If only the dataframe is passed, the function will
    do a 'count' and 'sum' of all columns. See also info().

    %piper sample_sales() >> summarise()

    | names         |   n |        sum |
    |:--------------|----:|-----------:|
    | location      | 200 |        nan |
    | product       | 200 |        nan |
    | month         | 200 |        nan |
    | target_sales  | 200 | 5423715.00 |
    | target_profit | 200 |  487476.42 |
    | actual_sales  | 200 | 5376206.33 |
    | actual_profit | 200 |  482133.92 |

    If you pass a groupby object to summarise() with no other
    parameters - summarise will summate all numeric columns.

    The equivalent to:
    %pipe df >> group_by(['col1', 'col2']) >> summarise(sum)


    Syntax examples
    ---------------
    # Syntax 1: column_name = ('existing_column', function)
    # note: below there are some 'common' functions that can
    # be quoted like 'sum', 'mean', 'count', 'nunique' or
    # just state the function name

    %%piper
    sample_sales() >>
    group_by('product') >>
    summarise(totval1=('target_sales', sum),
              totval2=('actual_sales', 'sum'))


    # Syntax 2: column_name = pd.NamedAgg('existing_column', function)
    %%piper
    sample_sales() >>
    group_by('product') >>
    summarise(totval1=(pd.NamedAgg('target_sales', 'sum')),
              totval2=(pd.NamedAgg('actual_sales', 'sum')))


    # Syntax 3: {'existing_column': function}
                {'existing_column': [function1, function2]}
    %%piper
    sample_sales()
    >> group_by('product')
    >> summarise({'target_sales':['sum', 'mean']})

    # Syntax 4: 'existing_column': lambda x: x+1
    # Example below identifies unique products sold by location.
    %%piper
    sample_sales() >>
    group_by('location') >>
    summarise({'product': lambda x: set(x.tolist())}) >>
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


# stack() {{{1
def stack(df: pd.DataFrame,
          *args,
          **kwargs) -> pd.DataFrame:
    ''' <*> stack dataframe

    This is a wrapper function rather than using e.g. df.stack()
    For details of args, kwargs - see help(pd.DataFrame.stack)


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    return df.stack(*args, **kwargs)


# tail() {{{1
def tail(df: pd.DataFrame,
         n: int = 4,
         shape: bool = True) -> pd.DataFrame:
    ''' <*> show last n records
    Like the corresponding R function, displays the last n records
    Alternative to df.tail().

    Usage example
    -------------
    ```python
    tail(df)
    ```


    Parameters
    ----------
    df : dataframe

    n : number of rows to display, default n=4

    shape: True, show shape information


    Returns
    -------
    A pandas dataframe
    '''
    if shape:
        _shape(df)

    return df.tail(n=n)


# to_csv() {{{1
def to_csv(df: pd.DataFrame,
          *args,
          **kwargs) -> None:
    ''' <*> to CSV

    This is a wrapper function rather than using e.g. df.to_csv()
    For details of args, kwargs - see help(pd.DataFrame.to_csv)


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''

    return df.to_csv(*args, **kwargs, index=False)


# to_tsv() {{{1
def to_tsv(df: pd.DataFrame,
           file_name: str,
           sep='\t') -> None:
    ''' <*> to TSV

    This is a wrapper function rather than using e.g. df.to_tsv()
    For details of args, kwargs - see help(pd.DataFrame.to_tsv)


    Parameters
    ----------
    df: dataframe

    file_name: output filename (extension assumed to be .tsv)

    sep: separator - default \t (tab-delimitted)


    Returns
    -------
    A pandas DataFrame
    '''
    file_name = _file_with_ext(file_name, extension='.tsv')

    rows = df.shape[0]
    if rows > 0:
        df.to_csv(file_name, sep=sep, index=False)
        logger.info(f'{file_name} exported, {df.shape[0]} rows.')


# to_parquet() {{{1
def to_parquet(df: pd.DataFrame, *args, **kwargs) -> None:
    ''' <*> to parquet

    This is a wrapper function rather than using e.g. df.to_parquet()
    For details of args, kwargs - see help(pd.DataFrame.to_parquet)


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''

    return df.to_parquet(*args, **kwargs)


# to_excel() {{{1
def to_excel(df: pd.DataFrame,
             file_name: str = None,
             *args,
             **kwargs) -> None:
    ''' <*> to excel

    This is a wrapper function for piper WorkBook class
    For details of args, kwargs - see help(piper.xl.WorkBook)


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    kwargs['filename'] = file_name
    kwargs['sheets'] = df

    WorkBook(*args, **kwargs)


# transform() {{{1
def transform(df: pd.DataFrame,
              index: Union[str, List[str]],
              **kwargs) -> pd.DataFrame:
    ''' <*> Add a group calculation to grouped DataFrame

    Transform is based on the pandas pd.DataFrame.transform() function.
    For details of args, kwargs - see help(pd.DataFrame.transform)

    Based on the given dataframe and grouping index, creates new aggregated column
    values using the list of keyword/value arguments (kwargs) supplied.

    By default, it calculates the group % of the first numeric column, in
    proportion to the first 'index' or grouping value.

    Usage example
    -------------
    Calculate group percentage value

    ```python
    %%piper
    sample_data() >>
    group_by(['countries', 'regions']) >>
    summarise(TotSales1=('values_1', 'sum'))
    ```

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

    ```python
    %%piper
    sample_data() >>
    group_by(['countries', 'regions']) >>
    summarise(TotSales1=('values_1', 'sum')) >>
    transform(index='countries', g_percent=('TotSales1', 'percent')) >>
    head(8)
    ```

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


# trim() {{{1
def trim(df: pd.DataFrame, str_columns: list = None) -> pd.DataFrame:
    ''' <*> strip leading/trailing blanks

    Parameters
    ----------
    df : pandas dataframe

    str_columns : Optional list of columns to strip.
                  If None, all string data type columns will be trimmed.

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
    ''' <*> unstack dataframe

    This is a wrapper function rather than using e.g. df.unstack()
    For details of args, kwargs - see help(pd.DataFrame.unstack)


    Parameters
    ----------
    df: dataframe

    *args: arguments for wrapped function

    **kwargs: keyword-parameters for wrapped function


    Returns
    -------
    A pandas DataFrame
    '''
    return df.unstack(*args, **kwargs)


# _set_grouper() {{{1
def _set_grouper(df: pd.DataFrame,
                index: Union[Any, List[Any]],
                freq: str = 'D') -> pd.DataFrame:
    ''' <*> Convert index to grouper index

    For given dataframe and index (str, list), return a dataframe with
    a pd.Grouper object for each defined index.


    Parameters
    ----------
    df: dataframe

    index: index column(s) to be converted to pd.Grouper objects

    freq: Default 'd' (days)
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


# _inplace() {{{1
def _inplace(df: pd.DataFrame, inplace: int = False) -> pd.DataFrame:
    ''' <*> Return a copy of the dataframe

    Helper function to return either a copy of dataframe or a reference to the
    passed dataframe


    Parameters
    ----------
    df - pandas dataframe


    Returns
    -------
    A pandas dataframe
    '''
    if inplace:
        return df
    else:
        return df.copy(deep=True)


# _shape() {{{1
def _shape(df: pd.DataFrame) -> pd.DataFrame:
    ''' <*> Show shape information

    Parameters
    ----------
    df - pandas dataframe


    Returns
    -------
    A pandas dataframe
    '''
    if isinstance(df, pd.Series):
        logger.info(f'{df.shape[0]} rows')

    if isinstance(df, pd.DataFrame):
        logger.info(f'{df.shape[0]} rows, {df.shape[1]} columns')
