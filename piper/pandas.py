import pandas as pd
import numpy as np
import logging
from datetime import datetime
from piper.verbs import trim
from piper.verbs import clean_columns
from piper.io import list_files
from piper.decorators import shape
from functools import wraps
import cx_Oracle
import re

logger = logging.getLogger(__name__)


# read_sql() {{{1
def read_sql(sql, con=None, sql_info=False, trim_blanks=True,
             return_sql=False, info=True):
    ''' Custom pandas pd.read_sql wrapper function

    Example:
    --------
    df = read_sql(sql=sql, con=con)

    df = read_sql(sql=get_sql_text(sql, variables),
                  con=con, sql_info=True, info=False)

    Parameters
    ----------
    sql : sql text to be executed

    con : database connection

    sql_info : True, False
               display logged sql code executed

    trim_blanks : True, False
                  if True, strip all column (row) values of leading
                  and trailing blanks

    info : True (default), False
           if True, display additional logging information

    Returns
    -------
    returns dataframe and optionally executed sql code (text)
    '''
    try:
        df = pd.read_sql(sql, con=con)
        df = clean_columns(df)
    except cx_Oracle.DatabaseError as e:
        logger.info(e)

    if not sql_info:
        logger.debug(sql)
    else:
        logger.info(sql)

    if trim_blanks:
        trim(df, inplace=True)
        if info:
            msg = f'Warning: Dataframe strip_blanks = {trim_blanks}'
            logger.info(msg)

    if info:
        note = "<NOTE: get_dataframe(return_sql=True) returns rendered SQL>"
        logger.info(note)

        logger.info(f'{df.shape[0]} rows, {df.shape[1]} columns')

    return df


# read_csv() {{{1
@shape(debug=False)
def read_csv(file_name, sep=',', strip_blanks=True, clean_cols=True,
             encoding='latin-1', info=True):
    ''' pd.read_csv wrapper function

    Parameters
    ----------
    file_name : csv formatted file

    sep : column separator

    strip_blanks : True, False
                   if True, strip all column (row) values of leading
                   and trailing blanks

    clean_cols (bool): default True -> lowercase column names, replace
                        spaces with underscore ('_')

    encoding: (str) : default 'latin-1'

    info : True, False
           if True, display additional logging information

    Returns
    -------
    pandas dataframe
    '''
    logger.info(f'{file_name}')

    df = pd.read_csv(file_name, sep=sep, encoding=encoding)

    if strip_blanks:
        if info:
            logger.info(f'Warning: Dataframe strip_blanks = {strip_blanks}')

        df = trim(df)

    if clean_cols:
        df = clean_columns(df)

    return df


# read_csvs() {{{1
@shape(debug=False)
def read_csvs(source='inputs/', glob_pattern='*.csv', sep=',', strip_blanks=True,
              clean_cols=True, encoding='latin-1', include_file=False, info=False):
    ''' pd.read_csv wrapper function

    Parameters
    ----------
    source - (str) source folder containing csv text files
             (that is, files ending with '*.csv')

    glob_pattern : file suffix filter - default '*.csv'

    sep : column separator - default ','

    strip_blanks : True, False
                   if True, strip all column (row) values of leading
                   and trailing blanks

    clean_cols (bool): default True -> lowercase column names, replace
                        spaces with underscore ('_')

    encoding: (str) : default 'latin-1'

    include_file (str) : include filename in returned dataframe - default False

    info : True, False
           if True, display additional logging information

    Returns
    -------
    pandas dataframe
    '''
    dataframes = []

    for file_ in list_files(source, glob_pattern=glob_pattern, as_posix=True):
        dx = pd.read_csv(file_, sep=sep)

        if strip_blanks:
            dx = trim(dx)
            if info:
                logger.info(f'Warning: Dataframe strip_blanks = {strip_blanks}')

        if clean_cols:
            dx = clean_columns(dx)

        if include_file:
            dx['filename'] = file_

        dataframes.append(dx)

    df = pd.concat(dataframes)

    return df


# read_excels() {{{1
def read_excels(source='inputs/',  glob_pattern='*.xls*', func=None,
                include_file=False, reset_index=True, info=False):
    '''
    Read, concatenate, combine and return a pandas dataframe based on
    workbook(s) data within a given source folder.

    Parameters
    ----------
    source - (str) source folder containing workbook data
             (that is, files ending with '.xls*')

    glob_pattern - file extension filter (str), default - '*.xls*'

    func - pass a custom function to read/transform each workbook/sheet.
            default is None (just performs a read_excel())

           Note: custom function will recieve Path object.
           To obtain the 'string' filename - use Path.as_posix()

    include_file (str) : include filename in returned dataframe - default False

    reset_index - (bool) reset index - default True

    info - (bool) - Provide debugging information - default False

    Returns
    -------
    pd.DataFrame - pandas DataFrame

    Example:
    -- Using a custom function, use below as a guide.

    def clean_data(xl_file):

        df = pd.read_excel(xl_file.as_posix(), header=None)

        # Combine first two rows to correct column headings
        df.columns = df.loc[0].values + ' ' + df.loc[1].values

        # Read remaining data and reference back to dataframe reference
        df = df.iloc[2:]

        # clean up column names to make it easier to use them in calculations
        df = clean_columns(df)

        # Set date data types for order and ship dates
        cols = ['order_date', 'ship_date']
        date_data = [pd.to_datetime(df[col]) for col in cols]
        df[cols] = pd.concat(date_data, axis=1)

        # Separate shipping mode from container
        df_split = df['ship_mode_container'].str.extract(r'(.*)-(.*)')
        df_split.columns = ['ship', 'container']

        priority_categories = ['Low', 'Medium', 'High', 'Critical', 'Not Specified']
        priority_categories = pd.CategoricalDtype(priority_categories, ordered=True)

        # split out and define calculated fields using lambdas
        sales_amount = lambda x: (x.order_quantity * x.unit_sell_price
                                  * (1 - x.discount_percent)).astype(float).round(2)

        days_to_ship=lambda x: ((x.ship_date - x.order_date) / pd.Timedelta(1, 'day')).astype(int)

        sales_person=lambda x: x.sales_person.str.extract('(Mr|Miss|Mrs) (\w+) (\w+)').loc[:, 1]

        order_priority=lambda x: x.order_priority.astype(priority_categories)

        # Define/assign new column (values)
        df = (df.assign(ship_mode=df_split.ship,
                        container=df_split.container,
                        sales_amount=sales_amount,
                        days_to_ship=days_to_ship,
                        sales_person=sales_person,
                        order_priority=order_priority)
                .drop(columns=['ship_mode_container'])
                .pipe(move_column, 'days_to_ship', 'after', 'ship_date')
                .pipe(move_column, 'sales_amount', 'after', 'unit_sell_price')
                .pipe(clean_columns, replace_char=('_', ' '), title=True)
             )

        return df

    data = read_excels('inputs/Data', func=clean_data)
    head(data, 2)

    '''

    dataframes = []
    for xl_file in list_files(source, glob_pattern=glob_pattern, as_posix=False):

        if func is None:
            df = pd.read_excel(xl_file)

        else:
            if info:
                logger.info(f'Using function {func.__name__}')

            df = func(xl_file)

        if info:
            logger.info(f'{xl_file}, {df.shape}')

        if include_file:
            df['filename'] = xl_file

        dataframes.append(df)

    data = pd.concat(dataframes)

    if reset_index:
        data = data.reset_index(drop=True)

    return data


# read_excel_sheets() {{{1
def read_excel_sheets(filename=None, sheets=None, include_sheet=False,
                      column='sheet_name', info=False, return_type='dataframe',
                      **kwargs):
    ''' For given Excel file name, return all or selected (list of) sheets as a
    consolidated pandas DataFrame

    Example
    -------
    dataframes = []

    for f in get_files(source='inputs/excel_workbooks/'):
        df = read_excel_sheets(f.as_posix(), include_sheet=True, column='category')
        df = df.assign(date = pd.to_datetime(f.stem, format='%m_%Y'))
        dataframes.append(df)

    df = pd.concat(dataframes)


    Parameters
    ----------
    filename - Excel Workbook (name) to be interrogated, default None

    sheets - list of selected sheets within Excel to be considered
             default None (include all sheets)

    include_sheet - include in returned dataframe a column containing sheet
                    name. Default False

    column - sheet column name, default 'sheet_name'

    **kwargs - keyword arguments to be passed to ExcelFile.parse() method

    info - print sheet names(s) read including rows, columns retrieved

    return_type - (str) options are:
                        'dataframe' - (default) return consolidated sheets in ONE dataframe
                        'dataframes' - return dictionary of sheet_name and sheet dataframe
                        'metadata' - return dataframe containing filename, sheet, rows, cols
                        'list' - return list of sheet names
    '''
    dataframes = []
    dataframe_dict = {}
    metadata = []

    with pd.ExcelFile(filename) as xl:

        for sheet in xl.sheet_names:

            if (sheets == None) or (sheet in sheets):
                dx = xl.parse(sheet, **kwargs)

                meta_dict = {'filename': filename, 'sheet_name': sheet,
                             'rows': dx.shape[0], 'columns': dx.shape[1]}
                metadata.append(meta_dict)

                if include_sheet:
                    dx[column] = sheet

                if info:
                    logger.info(f'sheet: {sheet}, (rows, cols) {dx.shape}')

                if return_type == 'dataframe':
                    dataframes.append(dx)
                elif return_type == 'dataframes':
                    dataframes_dict['sheet'] = dx

    if return_type == 'dataframe':
        df = pd.concat(dataframes)
        return df
    elif return_type == 'dataframes':
        return dataframes_dict
    elif return_type == 'metadata':
        df_meta = pd.DataFrame(metadata)
        return df_meta
    elif return_type == 'list':
        df_meta = pd.DataFrame(metadata)
        return df_meta['sheet_name'].tolist()


# split_dataframe() {{{1
def split_dataframe(df, chunk_size=1000):
    ''' Split dataframe by chunk_size rows, returning multiple dataframes

    1. Define 'range' (start, stop, step/chunksize)
    2. Use np.split() to examine dataframe indices using the calculated 'range'.

    Parameters
    ----------
    df - dataframe to be split

    chunksize - chunksize, default=1000

    Example
    -------
    chunks = split(customer_orders_tofix, 1000)
    for df in chunks:
        display(head(df, 2))

    returns
    -------
    A list of pd.DataFrame 'chunks'

    '''
    nrows = df.shape[0]
    range_ = range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)
    logger.debug(range_)

    return np.split(df, range_)


# paste() {{{1
def paste(source='text'):
    '''
    Convert clipboard sourced list from either a text list or from excel file

    columns = paste(source='text')

    Parameters
    ----------
    source : vertical        - copy from a vertical list of values (usually Excel)
             horizontal      - copy from a horizontal list of values (usually Excel)
             horizontal_list - return a horizontal list

             default 'vertical'
    '''
    if source == 'vertical':
        dx = pd.read_clipboard(header=None, names=['x'])
        data = "['" + "', '".join(dx.x.values.tolist()) + "']"

    if source == 'horizontal':
        dx = pd.read_clipboard(sep='\t')
        data = dx.columns.tolist()

    if source == 'horizontal_list':
        dx = pd.read_clipboard(sep='\t')
        data = "['" + "', '".join(dx.columns.tolist()) + "']"

    return data


# generate_summary_df() {{{1
def generate_summary_df(datasets, title='Summary', col_total='Total records',
                        add_grand_total=True, grand_total='Grand total'):
    ''' Given a dictionary of dataframes, transform into a summary
    of these dataframes with (optional) grand total row sum.

    Parameters
    ----------
    datasets : a list tuples containing dataframe description and dataframe(s)

    title : title (string) to be used in the summary dataframe.

    col_total : column total title (string)

    add_grand_total : True (default), False

    grand_total : grand total title (string)

    Returns
    -------
    pandas dataframe containing summary info

    Example:
    --------
    datasets = [('1st dataset', df), ('2nd dataset', df2)]
    summary_df = generate_summary_df(datasets, title='Summary',
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


# is_type() {{{1
def is_type(value, type_=str):
    ''' is type is useful when determining if a dataframe column
    contains a certain datatype (e.g. str, int, float)

    Parameters
    ----------
    value : value to be checked

    type : data type (e.g. str, int, float etc.)

    Example
    -------
    df[df.bgexdj.apply(is_type)]

    Returns
    -------
    boolean
    '''
    if isinstance(value, type_):
        return True

    return False
