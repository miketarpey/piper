from piper.sql import set_sql_text
from piper.sql import get_sql_text
from piper.text import _get_qual_file
from piper.sql import _get_sql_text
from piper.sql import set_sql_file
from os.path import join
from pathlib import Path

import pytest

directory = 'piper/temp/sql'

# get_config {{{1
@pytest.fixture
def get_config():

    config = {"meta": {"project": "Willow"},
              "folders": {"report": "Reports/Willow",
                          "sql": "sql/Cleanup - Inventory",
                          "input": "inputs",
                          "output": "outputs"},
              "variables": {"data_library": "wiseprdf",
                            "entity": "020",
                            "integer_value": 20,
                            "inventory_owner": "BAX101",
                            'list_values': ['2018', '2019']}}

    return config


# text_single_values() {{{1
@pytest.fixture
def text_single_values():

    text = ''' from {data_library}.invbal01
            left join {data_library}.wrkctr
              on hwareh = BWHSE and hdept = BDEPT
            left join {data_library}.status
              on hstat = zstats
            left join {data_library}.prthst05
              on hlent = phlent and hpart = phprtb
            where hcucc in ('{inventory_owner}')
            and integer = {integer_value} '''

    return text


# text_list_values() {{{1
@pytest.fixture
def text_list_values():

    text = ''' from {data_library}.invbal01
            left join {data_library}.wrkctr
              on hwareh = BWHSE and hdept = BDEPT
            left join {data_library}.status
              on hstat = zstats
            left join {data_library}.prthst05
              on hlent = phlent and hpart = phprtb
            where hcucc in ({list_values}) '''

    return text

# create_sql_file() {{{1
@pytest.fixture
def create_sql_file():

    file_name = join(directory, 'temp_sql.sql')
    sql = ''' from {data_library}.invbal01
            left join {data_library}.wrkctr
              on hwareh = BWHSE and hdept = BDEPT
            left join {data_library}.status
              on hstat = zstats
            left join {data_library}.prthst05
              on hlent = phlent and hpart = phprtb
            where hcucc in ({list_values}) '''

    with open(file_name, 'w') as f:
        f.write(sql)

# create_sql_files() {{{1
@pytest.fixture
def create_sql_files():

    file_name = join(directory, 'input sql.sql')
    sql = ''' from {schema}.invbal01
            left join {schema}.wrkctr
              on hwareh = BWHSE and hdept = BDEPT
            left join {schema}.status
              on hstat = zstats
            left join {schema}.prthst05
              on hlent = phlent and hpart = phprtb
            where hcucc in ({list_values}) '''

    with open(file_name, 'w') as f:
        f.write(sql)

    file_name = join(directory, 'expected sql.sql')
    sql = ''' from eudta.invbal01
            left join eudta.wrkctr
              on hwareh = BWHSE and hdept = BDEPT
            left join eudta.status
              on hstat = zstats
            left join eudta.prthst05
              on hlent = phlent and hpart = phprtb
            where hcucc in ({list_values}) '''

    with open(file_name, 'w') as f:
        f.write(sql)
# test_get_sql_text_with_list_values(get_config,{{{1
def test_get_sql_text_with_list_values(get_config, text_list_values):

    expected = ''' from wiseprdf.invbal01
            left join wiseprdf.wrkctr
              on hwareh = BWHSE and hdept = BDEPT
            left join wiseprdf.status
              on hstat = zstats
            left join wiseprdf.prthst05
              on hlent = phlent and hpart = phprtb
            where hcucc in ('2018', '2019') '''

    variables = get_config.get('variables')
    actual = _get_sql_text(text_list_values, variables, debug=False)

    assert expected == actual


# test_get_sql_text_with_single_int_values(get_config,{{{1
def test_get_sql_text_with_single_int_values(get_config, text_single_values):

    expected = ''' from wiseprdf.invbal01
            left join wiseprdf.wrkctr
              on hwareh = BWHSE and hdept = BDEPT
            left join wiseprdf.status
              on hstat = zstats
            left join wiseprdf.prthst05
              on hlent = phlent and hpart = phprtb
            where hcucc in ('BAX101')
            and integer = 20 '''

    variables = get_config.get('variables')
    actual = _get_sql_text(text_single_values, variables, debug=False)

    assert expected == actual


# test_get_sql_text_with_file(get_config, create_sql_file) {{{1
def test_get_sql_text_with_file(get_config, create_sql_file):

    expected = ''' from wiseprdf.invbal01
            left join wiseprdf.wrkctr
              on hwareh = BWHSE and hdept = BDEPT
            left join wiseprdf.status
              on hstat = zstats
            left join wiseprdf.prthst05
              on hlent = phlent and hpart = phprtb
            where hcucc in ('2018', '2019') '''

    variables = get_config.get('variables')
    actual = get_sql_text(join(directory, 'temp_sql.sql'), variables, debug=False)

    assert expected == actual


# test_get_sql_text_with_text(get_config) {{{1
def test_get_sql_text_with_text(get_config):
    """test that exception is raised for missing file"""

    expected = ''' from wiseprdf.invbal01
            left join wiseprdf.wrkctr
              on hwareh = BWHSE and hdept = BDEPT
            left join wiseprdf.status
              on hstat = zstats
            left join wiseprdf.prthst05
              on hlent = phlent and hpart = phprtb
            where hcucc in ('2018', '2019') '''

    text = ''' from {data_library}.invbal01
            left join {data_library}.wrkctr
              on hwareh = BWHSE and hdept = BDEPT
            left join {data_library}.status
              on hstat = zstats
            left join {data_library}.prthst05
              on hlent = phlent and hpart = phprtb
            where hcucc in ({list_values}) '''

    variables = get_config.get('variables')
    actual = get_sql_text(text, variables, debug=False)
    assert expected == actual


# test_set_sql_text_with_manual_template_values() {{{1
def test_set_sql_text_with_manual_template_values():

    expected = ''' select * from wiseprdf.table where column = 'some value' '''
    sql = ''' select * from {schema}.table where column = 'some value' '''

    template_values = {'schema': 'wiseprdf'}
    actual = set_sql_text(sql=sql, template_values=template_values)
    assert expected == actual


# test_set_sql_text_with_auto_template_values() {{{1
def test_set_sql_text_with_auto_template_values():

    expected = ''' select * from {schema}.table where column = 'some value' '''
    sql = ''' select * from eudta.table where column = 'some value' '''

    actual = set_sql_text(sql=sql, template_values=True)
    assert expected == actual


# test_set_sql_file_with_manual_template_values(create_sql_files) {{{1
def test_set_sql_file_with_manual_template_values(create_sql_files):

    from_file = join(directory, 'input sql.sql')
    to_file = directory / Path('output sql.sql')
    create_sql_files

    template_values = {'schema': 'eudta'}
    set_sql_file(from_file, to_file, template_values)

    with open(join(directory, 'expected sql.sql'), 'r') as f:
        expected = f.read()

    with open(join(directory, 'output sql.sql'), 'r') as f:
        actual = f.read()

    assert expected == actual


# test_set_sql_file_not_found() {{{1
def test_set_sql_file_not_found():
    """test that exception is raised for missing file"""

    from_file = join(directory, 'temp/file not found.sql')
    to_file = directory / Path('output sql.sql')
    template_values = {'schema': 'eudta'}

    with pytest.raises(Exception):
        assert set_sql_file(from_file, to_file, template_values) is None


