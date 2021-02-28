import logging
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path
from glob import glob
import pandas as pd

from os import walk
from os.path import split, normpath, join, relpath, basename
from zipfile import ZipFile, ZIP_DEFLATED

logger = logging.getLogger(__name__)


# _get_qual_file() {{{1
def _get_qual_file(folder, file_name, ts_prefix=True):
    ''' Get qualified xl file name

    Example
    -------
    _get_qual_file(folder, file_name, ts_prefix=False)

    _get_qual_file(folder, file_name, ts_prefix='date')

    _get_qual_file(folder, file_name, ts_prefix='time')


    Parameters
    ----------
    folder : folder path (string)

    file_name : file name (string) to create.

    ts_prefix : False (no timestamp)
                True (timestamp prefix)
                'date' (date only)

    Returns
    -------
    qual_file : (string) qualified file name
    '''
    if ts_prefix == 'date':
        ts = "{:%Y%m%d_}".format(datetime.now())
        qual_file = normpath(join(folder, ts + file_name))
    elif ts_prefix:
        ts = "{:%Y%m%d_%H%M}_".format(datetime.now())
        qual_file = normpath(join(folder, ts + file_name))
    else:
        qual_file = normpath(join(folder, file_name))

    return qual_file


# _file_with_ext() {{{1
def _file_with_ext(filename, extension='.xlsx'):
    ''' For given filename, check if required extension present.
        If not, append and return.
    '''
    if re.search(extension+'$', filename) is None:
        return filename + extension

    return filename


# read_text() {{{1
def read_text(file_name, count=True):
    '''
    read and return text/string data from a text file

    Parameters
    ----------

    file_name : file name (string)

    count : return row/record count only (default)

    Returns
    -------
    text : if count is True, return count (int)
           otherwise, return text data from file name (string)

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

# write_text() {{{1
def write_text(file_name, text):
    '''
    write text/string data to text file

    Parameters
    ----------

    file_name : file name (string)

    text : text string to be written to file name (string) to create.

    Returns
    -------
    None
    '''
    with open(file_name, 'w') as f:
        f.write(text)

# zip_data() {{{1
def zip_data(source='outputs', filter='*.xls*',
             target='outputs/zip_data', ts_prefix='date',
             include_folder=True, recurse=False,
             info=False, mode='w', test_mode=False):
    '''
    For given file or list of files, compress to target zip file provided.

    Example
    -------
    zip_data(source='outputs/', filter=f'{date_prefix}*.xsv',
             target='outputs/test', ts_prefix=True,
             include_folder=True, recurse=True, info=False,
             test_mode=False)

    Source: outputs, filter: 20200614*.xsv
    WARNING: No files found for selected folder/filter.

    Parameters
    ----------
    source : (str) source folder containing the files to zip

    filter : (str) default='*.xlsx', file filter

    target: (str) target zip file name to create.

    ts_prefix : (str/bool) - timestamp file name prefix.
                'date' (date only) -> default
                False (no timestamp)
                True (timestamp prefix)

    include_folder : True (default), False
                     True -> zipped files include the original folder
                     reference that the file resided in.

    recurse : (bool) - default (False) Recurse folders

    mode : (str) default 'w' (write), 'a' (add)

    test_mode : (bool) - default (False), if True - do not create zip

    Return
    ------
    None
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
        return

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
                logger.info(f'<<TEST MODE>> Target: {target_zip} not created with {idx} files.')
                return None
            else:
                logger.info(f'Target: {target_zip} created with {idx} files.')
                return zip


# list_files() {{{1
def list_files(source='inputs/', glob_pattern='*.xls*', recurse=False,
               as_posix=False, regex=''):
    ''' For a given source folder and file selection criteria, return
    a list of files. Criteria parameter allows one to focus on one
    or a group of files.

    Example
    -------
    List *ALL* files in /inputs directory, returning a list of paths:

    list_files(glob_pattern = '*', regex='Test', as_posix=True)

    Parameters
    ----------

    source - source folder (str), default - 'inputs/'

    glob_pattern - file extension filter (str), default - '*.xls*'

    regex - if specified, allows regular expression to further filter
            the file selection. Default is ''

            Example::
            list_files(glob_pattern = '*.tsv', regex='Test', as_posix=True)
            >['inputs/Test XL WorkBook.tsv']

    recurse - recurse directory (boolean), default False

    as_posix - if True, return list of files strings, default False

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


def duplicate_files(source=None, glob_pattern='*.*', recurse=False, filesize=1,
                   keep=False, xl_file=None):
    ''' For given source directory and global pattern (filter), all
    select files that have the same file size. This files are are assumed
    to be 'duplicates'.

    Example
    -------
    source = '/home/mike/Documents'
    duplicate_files(source, glob_pattern='*.*', recurse=True,
                filesize=2000000, keep=False).query("duplicate == True")

    References:
    -----------
    https://docs.python.org/3/library/pathlib.html
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html


    Parameters:
    -----------
    source : str - source directory, default None

    glob_pattern : str - filter extension suffix, default '*.*'

    recurse: bool - default False, if True, recurse source directory provided

    filesize : int - file size filter, default 1 (kb)

    keep: str - {‘first’, ‘last’, False}, default ‘first’
                Determines which duplicates (if any) to mark.
                    first : Mark duplicates as True except for the first occurrence.
                    last : Mark duplicates as True except for the last occurrence.
                    False : Mark all duplicates as True.

    xl_file : str - default None: output results to Excel workbook to xl_file


    Return:
    -------
    pd.DataFrame

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
