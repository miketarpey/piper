import os
import cx_Oracle
import logging
import numpy as np
import pandas as pd
import re
import zipfile
import logging
from datetime import datetime
from glob import glob
from os.path import split, normpath, join, relpath, basename
from pathlib import Path
from piper.decorators import shape
from piper.text import _file_with_ext
from piper.text import _get_qual_file
from piper.verbs import clean_columns
from piper.verbs import str_trim
from zipfile import ZipFile, ZIP_DEFLATED
from piper.xl import WorkBook

logger = logging.getLogger(__name__)


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


# duplicate_files() {{{1
def duplicate_files(source=None,
                   glob_pattern='*.*',
                   recurse=False,
                   filesize=1,
                   keep=False,
                   xl_file=None):
    ''' select files that have the same file size.

    This files are are assumed to be 'duplicates'.

    Parameters
    ----------
    source
        source directory, default None
    glob_pattern
        filter extension suffix, default '*.*'
    recurse
        default False, if True, recurse source directory provided
    filesize
        file size filter, default 1 (kb)
    keep
        {‘first’, ‘last’, False}, default ‘first’
        Determines which duplicates (if any) to mark.

        first: Mark duplicates as True except for the first occurrence.
        last: Mark duplicates as True except for the last occurrence.
        False: Mark all duplicates as True.
    xl_file
        default None: output results to Excel workbook to xl_file


    Returns
    -------
    pd.DataFrame


    Examples
    --------

    .. code-block::

        from piper.io import duplicate_files

        source = '/home/mike/Documents'

        duplicate_files(source,
                        glob_pattern='*.*',
                        recurse=True,
                        filesize=2000000,
                        keep=False).query("duplicate == True")

    **References**
    https://docs.python.org/3/library/pathlib.html
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html
    '''
    def func(f):

        try:
            size = os.stat(f.as_posix()).st_size
        except OSError as e:
            size = 0

        data = {'parent': f.parents[0].as_posix(),
                'name': f.name,
                # 'stem': f.stem, 'suffix': f.suffix,
                'size': size}

        return data


    file_data = list_files(source=source, glob_pattern = glob_pattern,
                           recurse=recurse, regex=None)

    file_data = [func(x) for x in file_data]

    df = (pd.DataFrame(file_data)
            .assign(duplicate = lambda x: x['size'].duplicated(keep=keep)))

    if filesize is not None:
        df = df.query("size >= @filesize")

    df = (df.sort_values(['size', 'name'], ascending=[False, True])
            .reset_index(drop=True))

    if xl_file is not None:
        logger.info(f'{xl_file} written, {df.shape[0]} rows')
        df.to_excel(xl_file, index=False)

    return df
# list_files() {{{1
def list_files(source='inputs/', glob_pattern='*.xls*', recurse=False,
               as_posix=False, regex=''):
    ''' return a list of files.

    Criteria parameter allows one to focus on one or a group of files.


    Examples
    --------
    List *ALL* files in /inputs directory, returning a list of paths:

    list_files(glob_pattern = '*', regex='Test', as_posix=True)


    Parameters
    ----------
    source
        source folder, default - 'inputs/'
    glob_pattern
        file extension filter (str), default - '*.xls*'
    regex
        if specified, allows regular expression to further filter the file
        selection. Default is ''

        *example*

        .. code-block::

            list_files(glob_pattern = '*.tsv', regex='Test', as_posix=True)
            >['inputs/Test XL WorkBook.tsv']
    recurse
        recurse directory (boolean), default False
    as_posix
        If True, return list of files strings, default False

    '''
    if glob_pattern in ('', None):
        raise ValueError(f'criteria {glob_pattern} value is invalid')

    files = list(Path(source).glob(glob_pattern))

    if recurse:
        files = list(Path(source).rglob(glob_pattern))

    if as_posix:
        files = [x.as_posix() for x in files]

    if regex not in (False, None):
        regexp = re.compile(regex)
        if as_posix:
            files = list(filter(lambda x: regexp.search(x), files))
        else:
            files = list(filter(lambda x: regexp.search(x.as_posix()), files))

    return files


# read_csv() {{{1
@shape(debug=False)
def read_csv(file_name: str,
             sep: str = ',',
             strip_blanks: bool = True,
             clean_cols: bool = True,
             encoding: str = None,
             info: bool = True) -> pd.DataFrame:
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
        default None. Tip:: For EU data try 'latin-1'
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
        str_trim(df)
        if info:
            msg = f'Warning: Dataframe strip_blanks = {trim_blanks}'
            logger.info(msg)

    if info:
        note = "<NOTE: get_dataframe(return_sql=True) returns rendered SQL>"
        logger.info(note)

        logger.info(f'{df.shape[0]} rows, {df.shape[1]} columns')

    return df


# read_text() {{{1
def read_text(file_name, count=True):
    ''' read and return text/string data from a text file

    Parameters
    ----------

    file_name
        file name (string)
    count
        return row/record count only (default)

    Returns
    -------
    text
        if count is True, return count (int) otherwise, return text data from
        file name (string)

    '''
    with open(file_name, 'r') as f:
        text_data = f.readlines()

    total_records = len(text_data)

    logger.info(f'{file_name}')
    logger.info(f'{total_records} rows')

    if count:
        return total_records
    else:
        return text_data


# to_csv() {{{1
def to_csv(df: pd.DataFrame,
          *args,
          **kwargs) -> None:
    '''to CSV

    This is a wrapper function rather than using e.g. df.to_csv()
    For details of args, kwargs - see help(pd.DataFrame.to_csv)


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

    return df.to_csv(*args, **kwargs, index=False)


# to_tsv() {{{1
def to_tsv(df: pd.DataFrame,
           file_name: str,
           sep='\t') -> None:
    '''to TSV

    This is a wrapper function rather than using e.g. df.to_tsv()
    For details of args, kwargs - see help(pd.DataFrame.to_tsv)


    Parameters
    ----------
    df
        dataframe
    file_name
        output filename (extension assumed to be .tsv)
    sep
        separator - default \t (tab-delimitted)


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
    '''to parquet

    This is a wrapper function rather than using e.g. df.to_parquet()
    For details of args, kwargs - see help(pd.DataFrame.to_parquet)


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

    return df.to_parquet(*args, **kwargs)


# to_excel() {{{1
def to_excel(df: pd.DataFrame,
             file_name: str = None,
             *args,
             **kwargs) -> None:
    '''to excel

    This is a wrapper function for piper WorkBook class
    For details of args, kwargs - see help(piper.xl.WorkBook)


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
    kwargs['file_name'] = file_name
    kwargs['sheets'] = df

    WorkBook(*args, **kwargs)


# write_text() {{{1
def write_text(file_name, text):
    ''' write text/string data to text file

    Parameters
    ----------

    file_name
        file name (string)
    text
        text string to be written to file name (string) to create.

    Returns
    -------
    None
    '''
    with open(file_name, 'w') as f:
        f.write(text)

# zip_data() {{{1
def zip_data(source: str = 'outputs',
             filter: str = '*.xls*',
             target: str = 'outputs/zip_data',
             ts_prefix: str = 'date',
             include_folder: bool = True,
             recurse: bool = False,
             info: bool = False,
             mode: str = 'w',
             test_mode: bool = False) -> Union[ZipFile, Any]:
    ''' Compress files from source to target folder

    Parameters
    ----------
    source
        source folder containing the files to zip.
    filter
        default scans for xlsx files.
    target
        target zip file name to create.
    ts_prefix
        timestamp file name prefix.
        'date' (date only) -> default
        False (no timestamp)
        True (timestamp prefix)
    include_folder
        True (default). Zipped files include the original folder reference that
        the file resided in.
    recurse
        default (False) Recurse folders
    mode
        default 'w' (write), 'a' (add)
    test_mode
        default (False), if True - do not create zip


    Returns
    -------
    None


    Examples
    --------

    .. code-block::

        zip_data(source='outputs/', filter=f'{date_prefix}*.xsv',
                 target='outputs/test', ts_prefix=True,
                 include_folder=True, recurse=True, info=False,
                 test_mode=False)

    Source: outputs, filter: 20200614*.xsv
    WARNING: No files found for selected folder/filter.

    '''
    source_folder, target_zip = Path(source), Path(target)

    target_zip = _get_qual_file(target_zip.parent, target_zip.name,
                                ts_prefix=ts_prefix)

    target_zip = Path(_file_with_ext(target_zip, extension='.zip'))
    logger.info(f'Source: {source_folder}, filter: {filter}')

    if recurse:
        file_list = source_folder.rglob(filter)
    else:
        file_list = source_folder.glob(filter)

    files_to_zip = list(file_list)
    if len(files_to_zip) == 0:
        logger.info('WARNING: No files found for selected folder/filter.')
        return None

    if isinstance(source_folder, Path):
        with ZipFile(target_zip, mode=mode, compression=ZIP_DEFLATED) as zip:

            for idx, f in enumerate(files_to_zip, start=1):

                if not test_mode:
                    if include_folder:
                        zip.write(filename=f)
                    else:
                        zip.write(filename=f, arcname=f.name)

                if info:
                    logger.info(f'file: {f} written')

            if test_mode:
                msg = f'<TEST> Target: {target_zip} - {idx} files not created.'
                logger.info(msg)
            else:
                logger.info(f'Target: {target_zip} - {idx} files created.')

            return zip
