import pandas as pd
import numpy as np
import logging
from datetime import datetime
from piper.verbs import str_trim
from piper.verbs import clean_columns
from piper.io import list_files
from piper.decorators import shape
import cx_Oracle
import re

logger = logging.getLogger(__name__)


# read_sql() {{{1
def read_sql(sql,
             con = None,
             sql_info = False,
             trim_blanks = True,
             return_sql = False,
             info = True):
    ''' Custom pandas pd.read_sql wrapper function

    Examples
    --------
    df = read_sql(sql=sql, con=con)

    df = read_sql(sql=get_sql_text(sql, variables),
                  con=con, sql_info=True, info=False)

    Parameters
    ----------
    sql
        sql text to be executed
    con
        database connection
    sql_info
        Default False
        If True, display logged sql code executed
    trim_blanks
        If true, strip all column (row) values of leading and trailing blanks
    info
        Default True
        If True, display additional logging information

    Returns
    -------
    a pandas dataframe and optionally executed sql code (text)
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
        str_trim(df, inplace=True)
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
def read_csv(file_name,
             sep = ',',
             strip_blanks = True,
             clean_cols = True,
             encoding = 'latin-1',
             info = True):
    ''' pd.read_csv wrapper function

    Parameters
    ----------
    file_name
        csv formatted file
    sep
        column separator
    strip_blanks
        Default True
        If True, strip all column (row) values of leading and trailing blanks
    clean_cols
        default True
        If True, lowercase column names, replace spaces with underscore ('_')
    encoding
        default 'latin-1'
    info
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

        df = str_trim(df)

    if clean_cols:
        df = clean_columns(df)

    return df


# read_csvs() {{{1
@shape(debug=False)
def read_csvs(source = 'inputs/',
              glob_pattern = '*.csv',
              sep = ',',
              strip_blanks = True,
              clean_cols = True,
              encoding = 'latin-1',
              include_file = False,
              info = False):
    '''pd.read_csv wrapper function

    Parameters
    ----------
    source
        source folder containing csv text files (that is, files ending with '\*.csv')
    glob_pattern
        global pattern for file selection
    sep
        Column separator
    strip_blanks
        Default True
        If True, strip all column (row) values of leading and trailing blanks
    clean_cols
        default True. Lowercase column names, replace spaces with underscore ('\_')
    encoding
        default 'latin-1'
    include_file
        include filename in returned dataframe - default False
    info
        if True, display additional logging information

    Returns
    -------
    pandas dataframe
    '''
    dataframes = []

    for file_ in list_files(source, glob_pattern=glob_pattern, as_posix=False):
        dx = pd.read_csv(file_.as_posix(), sep=sep)

        if strip_blanks:
            dx = str_trim(dx)
            if info:
                logger.info(f'Warning: Dataframe strip_blanks = {strip_blanks}')

        if clean_cols:
            dx = clean_columns(dx)

        if include_file:
            dx['filename'] = file_.name

        dataframes.append(dx)

    df = pd.concat(dataframes)

    return df


# read_excels() {{{1
def read_excels(source = 'inputs/',
                glob_pattern = '*.xls*',
                func = None,
                include_file = False,
                reset_index = True,
                info = False):
    '''
    Read, concatenate, combine and return a pandas dataframe based on
    workbook(s) data within a given source folder.

    Parameters
    ----------
    source
        source folder containing workbook data (that is, files ending with '.xls*')
    glob_pattern
        file extension filter (str), default - '*.xls*'
    func
        pass a custom function to read/transform each workbook/sheet. default is
        None (just performs a read_excel())

        Note: custom function will recieve Path object. To obtain the 'string'
        filename - use Path.as_posix()
    include_file
        include filename in returned dataframe - default False
    reset_index
        default True
    info
        Provide debugging information - default False


    Returns
    -------
    pd.DataFrame - pandas DataFrame


    Examples
    --------
    Using a custom function, use below as a guide.

    .. code-block::

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
    for xl_file in list_files(source,
                              glob_pattern=glob_pattern,
                              as_posix=False):

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
def read_excel_sheets(filename = None,
                      sheets = None,
                      include_sheet = False,
                      column = 'sheet_name',
                      info = False,
                      return_type = 'list_frames',
                      **kwargs):
    ''' For given Excel file name, return all or selected (list of) sheets as a
    consolidated pandas DataFrame

    Examples
    --------
    test1, test2 = read_excel_sheets(xl_file, return_type='list_frames')


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

                        'list_frames' - (default) return a list of dataframes

                        'list' - return list of sheet names

                        'dict_frames' - return a dictionary containing:
                                        sheet_name: dataframe

                        'combine' - return consolidated sheets in ONE dataframe

                        'metadata' - return dataframe containing filename, sheet, rows, cols

    '''
    dataframes = []
    dataframes_dict = {}
    metadata = []

    with pd.ExcelFile(filename) as xl:

        for sheet in xl.sheet_names:

            if (sheets is None) or (sheet in sheets):
                dx = xl.parse(sheet, **kwargs)

                meta_dict = {'filename': filename, 'sheet_name': sheet,
                             'rows': dx.shape[0], 'columns': dx.shape[1]}
                metadata.append(meta_dict)

                if include_sheet:
                    dx[column] = sheet

                if info:
                    logger.info(f'sheet: {sheet}, (rows, cols) {dx.shape}')

                if return_type == 'list_frames' or \
                   return_type == 'combine':
                    dataframes.append(dx)

                elif return_type == 'dataframes':
                    dataframes_dict['sheet'] = dx

    if return_type == 'list_frames':
        return dataframes
    elif return_type == 'list':
        df_meta = pd.DataFrame(metadata)
        return df_meta['sheet_name'].tolist()
    elif return_type == 'dict_frames':
        return dataframes_dict
    if return_type == 'combine':
        df = pd.concat(dataframes)
        return df
    elif return_type == 'metadata':
        df_meta = pd.DataFrame(metadata)
        return df_meta


