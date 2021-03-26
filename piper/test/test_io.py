from piper.io import list_files
from piper.io import read_text
from piper.io import write_text
from piper.io import zip_data

import pytest

directory = 'piper/temp/sql'

@pytest.fixture
def sql():
    return ''' select * from eudta.f0006 where rownum=1'''


# test_read_sql_valid_info_true {{{1
@pytest.mark.skip(reason="no way of currently testing this")
def test_read_sql_valid_info_true(sql):
    """
    """
    env = 'JDE9E2P'
    con, schema, schema_ctl = connect(env)
    df = read_sql(sql, sql_info=True, con=con, info=False)
    assert (1, 152) == df.shape


# test_read_sql_valid_info_false
@pytest.mark.skip(reason="no way of currently testing this")
def test_read_sql_valid_info_false(sql):
    """
    """
    env = 'JDE9E2P'
    con, schema, schema_ctl = connect(env)
    df = read_sql(sql, sql_info=False, con=con, info=True)
    assert (1, 152) == df.shape

# write_text_file {{{1
@pytest.fixture
def write_text_file():
    ''' write sample text file '''

    filename = 'piper/temp/sample_text_file.txt'
    text = 'some sample text'
    write_text(filename, text)


# test_list_files_no_files() {{{1
def test_list_files_no_files():

    source = 'piper/'

    test_files = list_files(source=source, glob_pattern='*.px')

    expected = []
    actual = test_files

    assert expected == actual


# test_list_files_with_data() {{{1
def test_list_files_with_data():

    source = 'piper/'

    test_files = list_files(source=source, glob_pattern='*.py')

    expected = 17
    actual = len(test_files)

    assert expected == actual


# test_list_files_with_data_as_posix() {{{1
def test_list_files_with_data_as_posix():

    source = 'piper/'

    test_files = list_files(source=source, glob_pattern='*.py',
                            as_posix=True)

    expected = True
    actual = isinstance(test_files[0], str)

    assert expected == actual


# test_read_text(write_text_file) {{{1
def test_read_text(write_text_file):

    filename = 'piper/temp/sample_text_file.txt'
    expected = ['some sample text']

    assert expected == read_text(filename, count=False)


# test_zip_data(write_text_file) {{{1
def test_zip_data(write_text_file):

    filename = 'piper/temp/test_zip_file'

    test_zip = zip_data(source='piper/temp', target=filename,
                        filter='*.txt', ts_prefix=False,
                        test_mode=False, mode='w', info=False)

    expected = 1
    actual = len(test_zip.namelist())

    assert expected == actual


# test_zip_data_test_mode(write_text_file) {{{1
def test_zip_data_test_mode(write_text_file):

    filename = 'piper/temp/test_zip_file'

    test_zip = zip_data(source='temp', target=filename,
                        filter='*.sql', ts_prefix=False,
                        test_mode=True, mode='w', info=False)

    expected = None
    actual = test_zip

    assert expected == actual

# test_zip_data_nofiles(write_text_file) {{{1
@pytest.fixture
def test_zip_data_nofiles(write_text_file):

    filename = 'piper/temp/test_zip_file'

    test_zip = zip_data(source='temp', target=filename,
                        filter='*.nofiles', ts_prefix=False,
                        test_mode=False, mode='w', info=False)

    expected = None
    actual = test_zip

    assert expected == actual
