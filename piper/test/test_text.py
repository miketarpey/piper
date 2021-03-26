from datetime import datetime
from pathlib import Path
from os.path import join
from piper.text import pipe_parser
from piper.text import pipe_function_parms
from piper.text import set_sql_text
from piper.text import get_sql_text
from piper.text import _get_qual_file
from piper.text import _get_sql_text

import pytest

directory = 'piper/temp/sql'

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


# test_parser_single_statement {{{1
def test_parser_single_statement():
    """
    """
    text = '''
    get_sample_data(end='2020')
    '''

    expected = """(get_sample_data(end='2020'))"""

    actual, call_function = pipe_parser(text, debug=False)

    assert expected == actual


# test_parser_two_statements {{{1
def test_parser_two_statements():
    """
    """
    text = ''' get_sample_data(end='2020') >> where("values_1 > 200")'''

    expected = """(get_sample_data(end='2020')\n.pipe(where, "values_1 > 200"))"""

    actual, call_function = pipe_parser(text, debug=False)

    assert expected == actual


# test_get_qualified_file() {{{1
def test_get_qualified_file():

    ts = "{:%Y%m%d_}".format(datetime.now())
    ts2 = "{:%Y%m%d_%H%M}_".format(datetime.now())

    folder = 'input'
    file_name = 'some arbitrary file name.xlsx'

    expected = 'input/some arbitrary file name.xlsx'
    actual = _get_qual_file(folder, file_name, ts_prefix=False)
    assert expected == actual

    expected = f'input/{ts}some arbitrary file name.xlsx'
    actual = _get_qual_file(folder, file_name, ts_prefix='date')
    assert expected == actual

    expected = f'input/{ts2}some arbitrary file name.xlsx'
    actual = _get_qual_file(folder, file_name, ts_prefix='time')
    assert expected == actual


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


# test_pipe_function_parms_single_statement {{{1
def test_pipe_function_parms_single_statement():

    statements = [('get_sample_sales(', ')')]

    expected = ['get_sample_sales()']
    actual = [pipe_function_parms(idx, x) for idx, x in enumerate(statements)]

    assert expected == actual


# test_pipe_function_parms_no_parms {{{1
def test_pipe_function_parms_no_parms():

    statements = [('get_sample_sales(', ')'),
                  ('select(', ')'),
                  ('head(', ')')]

    expected = ['get_sample_sales()',
                '.pipe(select)', '.pipe(head)']

    actual = [pipe_function_parms(idx, x) for idx, x in enumerate(statements)]

    assert expected == actual


# test_pipe_function_parms_single_parm {{{1
def test_pipe_function_parms_single_parm():

    statements = [('get_sample_sales(', ')'),
                  ('select(', ')'),
                  ('head(', 'n=4)')]

    expected = ['get_sample_sales()',
                '.pipe(select)', '.pipe(head, n=4)']

    actual = [pipe_function_parms(idx, x) for idx, x in enumerate(statements)]

    assert expected == actual


# test_pipe_function_parms_multi_parms {{{1
def test_pipe_function_parms_multi_parms():

    statements = [('get_sample_sales(', ')'),
                  ('select(', "['-date', 'something_else'])"),
                  ('head(', 'n=4)')]

    expected = ['get_sample_sales()',
                ".pipe(select, ['-date', 'something_else'])",
                '.pipe(head, n=4)']

    actual = [pipe_function_parms(idx, x) for idx, x in enumerate(statements)]

    assert expected == actual
