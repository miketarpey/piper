import logging
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path
from glob import glob

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
def list_files(source='inputs/', filter='*.xls*', recurse=False, as_posix=False):
    ''' For given source folder and filter, return list of files

    Parameters
    ----------

    source - source folder (str), default - 'inputs/'

    filter - file extension filter (str), default - '*.xls*'

    recurse - recurse directory (boolean), default False

    as_posix - if True, return list of files as string names, default False

    '''
    files = Path(source)

    if recurse:
        files = list(files.rglob(filter))
    else:
        files = list(files.glob(filter))

    if as_posix:
        return [x.as_posix() for x in files]

    return files
