import pytest
import json
import pandas as pd
from piper.xl import WorkBook
from piper.factory import bad_quality_orders
from piper.factory import xl_test_data
from pathlib import Path


relative_folder = Path(__file__).parents[1] / 'temp/'


@pytest.fixture
def sample_orders_01():
    return bad_quality_orders()


@pytest.fixture
def sample_orders_02():
    return xl_test_data()


def test_workbook_add_sheet_auto(sample_orders_01):

    file_name = relative_folder / 'WorkBook - auto sheet.xlsx'
    df = pd.DataFrame(sample_orders_01)

    wb = WorkBook(file_name, ts_prefix=False)
    wb.add_sheet(df, sheet_name='**auto')
    wb.close()

    expected = 1
    actual = wb.last_sheet_idx

    assert expected == actual


def test_workbook_add_sheet_test_zoom(sample_orders_01):

    file_name = relative_folder / 'WorkBook - zoom.xlsx'
    df = pd.DataFrame(sample_orders_01)

    wb = WorkBook(file_name, ts_prefix=False)
    wb.add_sheet(df, sheet_name='**auto', zoom=130)
    wb.close()

    expected = 130
    actual = wb.sheet_dict.get('sheet1')[6]

    assert expected == actual


def test_workbook_add_sheet_test_tab_color(sample_orders_01):

    file_name = relative_folder / 'WorkBook - tab color.xlsx'
    df = pd.DataFrame(sample_orders_01)

    wb = WorkBook(file_name, ts_prefix=False)
    wb.add_sheet(df, sheet_name='**auto', tab_color='green')
    wb.close()

    expected = 'green'
    actual = wb.sheet_dict.get('sheet1')[5]

    assert expected == actual


def test_workbook_add_sheet_test_index(sample_orders_01):

    file_name = relative_folder / 'WorkBook - with index.xlsx'
    df = pd.DataFrame(sample_orders_01)

    wb = WorkBook(file_name, ts_prefix=False)
    wb.add_sheet(df, sheet_name='**auto', index=True)
    wb.close()

    expected = True
    actual = wb.sheet_dict.get('sheet1')[3]

    assert expected == actual


def test_workbook_add_sheet_test_invalid_theme(sample_orders_01):

    file_name = relative_folder / 'WorkBook - sheet dictionary meta.xlsx'
    df = pd.DataFrame(sample_orders_01)

    wb = WorkBook(file_name, ts_prefix=False)

    with pytest.raises(Exception):
        assert wb.add_sheet(df, sheet_name='**auto', theme='invalid') is None

    wb.close()


def test_workbook_add_sheet_test_sheet_dict(sample_orders_01):

    df = pd.DataFrame((sample_orders_01))

    file_name = relative_folder / 'WorkBook - sheet dictionary meta.xlsx'
    wb = WorkBook(file_name, ts_prefix=False)
    expected_ws_reference = wb.add_sheet(df, sheet_name='**auto')
    wb.close()

    expected_shape = (12, 9)
    actual = wb.sheet_dict.get('sheet1')[0]
    assert expected_shape == actual

    expected_index_names = df.index.names
    actual = wb.sheet_dict.get('sheet1')[1]
    assert expected_index_names == actual

    expected_header = True
    actual = wb.sheet_dict.get('sheet1')[2]
    assert expected_header == actual

    expected_index = False
    actual = wb.sheet_dict.get('sheet1')[3]
    assert expected_index == actual

    actual = wb.sheet_dict.get('sheet1')[4]
    assert expected_ws_reference == actual


def test_workbook_add_sheet_sql(sample_orders_01):

    df = pd.DataFrame((sample_orders_01))

    file_name = relative_folder / 'WorkBook - add SQL sheet.xlsx'
    wb = WorkBook(file_name, ts_prefix=False)

    sql = ''' select * from eudta.f56474z1 '''
    wb.add_sheet(df, sql=sql, sheet_name='**auto')
    wb.close()

    expected_shape = (1, 1)
    actual = wb.sheet_dict.get('_sheet1')[0]
    assert expected_shape == actual


def test_workbook_get_range_all(sample_orders_01):

    df = pd.DataFrame(sample_orders_01)

    file_name = relative_folder / 'WorkBook - get_range test.xlsx'
    wb = WorkBook(file_name, ts_prefix=False)

    wb.add_sheet(df, sheet_name='**auto')
    wb.close()

    expected_shape = '$A$1:$I$13'
    actual = wb.get_range(sheet_name='sheet1')
    assert expected_shape == actual


def test_workbook_get_range_single_column(sample_orders_01):

    df = pd.DataFrame(sample_orders_01)

    file_name = relative_folder / 'WorkBook - get_range test.xlsx'
    wb = WorkBook(file_name, ts_prefix=False)

    wb.add_sheet(df, sheet_name='**auto')
    wb.close()

    expected_shape = '$E$1:$E$13'
    actual = wb.get_range(sheet_name='sheet1',
                          column_range='E')
    assert expected_shape == actual


def test_workbook_get_range_from_to(sample_orders_01):

    df = pd.DataFrame(sample_orders_01)

    file_name = relative_folder / 'WorkBook - get_range test.xlsx'
    wb = WorkBook(file_name, ts_prefix=False)

    wb.add_sheet(df, sheet_name='**auto')
    wb.close()

    expected_shape = '$D$1:$F$13'
    actual = wb.get_range(sheet_name='sheet1',
                          column_range='D:F')
    assert expected_shape == actual


def test_workbook_add_sheet_conditional_fmt(sample_orders_02):

    df = pd.DataFrame(sample_orders_02)

    file_name = relative_folder / 'WorkBook - conditional fmt1.xlsx'
    wb = WorkBook(file_name, ts_prefix=False)

    sheet_name = 'conditional_test'
    ws = wb.add_sheet(df, sheet_name=sheet_name)

    c = {'type': 'duplicate', 'format': 'accent1', 'range': 'D:E'}
    wb.add_condition(ws, c)

    c = {'type': 'text', 'criteria': 'containing', 'value': 'Customer1',
            'format': 'accent2', 'range': 'B'}
    wb.add_condition(ws, c)

    c = {'type': 'formula', 'criteria': '=$H2=558.72', 'format': 'accent4'}
    wb.add_condition(ws, c)

    wb.close()

    expected_shape = (12, 9)
    actual = wb.sheet_dict.get(sheet_name)[0]
    assert expected_shape == actual


def test_get_metadata_invalid(sample_orders_01):

    file_name = relative_folder / 'WorkBook - test metadata.xlsx'
    df = pd.DataFrame(sample_orders_01)

    wb = WorkBook(file_name, ts_prefix=False)
    wb.add_sheet(df, sheet_name='**auto')

    with pytest.raises(Exception):
        assert wb.get_metadata('invalid_file.json')

    wb.close()


def test_get_styles_invalid(sample_orders_01):

    file_name = relative_folder / 'WorkBook - test metadata.xlsx'
    df = pd.DataFrame(sample_orders_01)

    wb = WorkBook(file_name, ts_prefix=False)
    wb.add_sheet(df, sheet_name='**auto')

    with pytest.raises(Exception):
        assert wb.get_styles('invalid_file.json')

    wb.close()


def test_WorkBook_Sheet_list_None_(sample_orders_01):

    file_name = relative_folder / 'WorkBook None sheet.xlsx'

    expected = WorkBook
    actual = type(WorkBook(file_name, sheets=None, ts_prefix=False))

    assert expected == actual


def test_WorkBook_Sheet_list_single_dataframe(sample_orders_01):

    df = pd.DataFrame(sample_orders_01)
    file_name = relative_folder / 'WorkBook Single dataframe.xlsx'
    WorkBook(file_name, sheets=df, ts_prefix=False)

    rows, columns = 12, 9
    actual = df.shape
    assert rows, columns == actual


def test_WorkBook_Sheet_list_list_of_dataframes(sample_orders_01, sample_orders_02):

    df = pd.DataFrame(sample_orders_01)
    df2 = pd.DataFrame(sample_orders_02)
    file_name = relative_folder / 'WorkBook list of dataframes.xlsx'
    WorkBook(file_name, sheets=[df, df2], ts_prefix=False)

    rows, columns = 12, 9
    actual = df2.shape
    assert rows, columns == actual


def test_WorkBook_Sheet_list_dictionary_of_dataframes(sample_orders_01,
                                                     sample_orders_02):

    df = pd.DataFrame(sample_orders_01)
    df2 = pd.DataFrame(sample_orders_02)
    file_name = relative_folder / 'WorkBook list of tuples.xlsx'
    WorkBook(file_name, sheets={'SheetA': df, 'SheetB': df2}, ts_prefix=False)

    rows, columns = 12, 9
    actual = df2.shape
    assert rows, columns == actual


def test_WorkBook_Sheet_list_number_of_rows(sample_orders_02):

    df = pd.DataFrame(sample_orders_02)
    file_name = relative_folder / 'WorkBook single - shape test.xlsx'
    WorkBook(file_name, sheets=df, ts_prefix=False)

    rows, columns = 12, 9
    actual = df.shape
    assert rows, columns == actual


def test_WorkBook_Sheet_add_format(sample_orders_01):
    ''' Not sure how I can 'test' add_format directly.
    '''
    df = sample_orders_01

    file_name = relative_folder / 'WorkBook - add_format.xlsx'
    wb = WorkBook(file_name, ts_prefix=False)

    ws = wb.add_sheet(df, sheet_name='original', tab_color='yellow', zoom=175)

    attr = {'column': 'D', 'format': 'center', 'width': 11}
    wb.add_format(ws, column_attr=attr)

    wb.close()


def test_WorkBook_Sheet_add_formats(sample_orders_01):
    ''' Not sure how I can 'test' add_format directly.
    '''
    df = sample_orders_01

    file_name = relative_folder / 'WorkBook - add_formats.xlsx'
    wb = WorkBook(file_name, ts_prefix=False)

    ws = wb.add_sheet(df, sheet_name='original', tab_color='yellow', zoom=175)

    centre_cols = ['C', 'D', 'E']

    formats = []
    for col in centre_cols:
        attr = {'column': col, 'format': 'center', 'width': 11}
        formats.append(attr)

    wb.add_format(ws, column_attr=formats)

    wb.close()


def test_WorkBook_Sheet_add_condition(sample_orders_01):
    ''' Not sure how I can 'test' add_condition directly.
    '''
    df = sample_orders_01

    file_name = relative_folder / 'WorkBook - add_condition.xlsx'
    wb = WorkBook(file_name, ts_prefix=False)

    ws = wb.add_sheet(df, sheet_name='original', tab_color='yellow', zoom=175)

    c = {'type': 'formula', 'criteria': '=$E2="B"', 'format': 'accent1', 'range': 'E'}
    wb.add_condition(ws, c)

    wb.close()


def test_WorkBook_Sheet_add_condition_default_error_format(sample_orders_01):
    ''' Not sure how I can 'test' add_condition directly.
    '''
    df = sample_orders_01

    file_name = relative_folder / 'WorkBook - add_condition - default error format.xlsx'
    wb = WorkBook(file_name, ts_prefix=False)

    ws = wb.add_sheet(df, sheet_name='original', tab_color='yellow', zoom=175)

    c = {'type': 'formula', 'criteria': '=$E2="B"',  'range': 'E'}
    wb.add_condition(ws, c)

    wb.close()


def test_WorkBook_Sheet_add_conditions(sample_orders_01):
    ''' Not sure how I can 'test' add_condition directly.
    '''
    df = sample_orders_01

    file_name = relative_folder / 'WorkBook - add_conditions.xlsx'
    wb = WorkBook(file_name, ts_prefix=False)

    ws = wb.add_sheet(df, sheet_name='original', tab_color='yellow', zoom=175)

    c = [ {'type': 'formula', 'criteria': '=$F2>$G2', 'format': 'accent4', 'range': 'F:G'},
          {'type': 'cell', 'criteria': 'equal to', 'value': '"Spain"', 'format': 'accent5', 'range': 'C'}]
    wb.add_condition(ws, c)

    wb.close()


def test_WorkBook_get_themes(sample_orders_01):
    '''
    '''
    file_name = relative_folder / 'WorkBook - test themes.xlsx'
    wb = WorkBook(file_name, ts_prefix=False)
    assert wb.show_themes() == ', '.join(wb.get_themes())
    wb.close()


def test_WorkBook_show_styles(sample_orders_01):
    '''
    '''
    file_name = relative_folder / 'WorkBook - test themes.xlsx'
    wb = WorkBook(file_name, ts_prefix=False)
    assert wb.show_styles() == wb.get_styles().keys()
    wb.close()
