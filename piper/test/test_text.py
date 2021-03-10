from piper.text import pipe_parser
from piper.text import join_function_parms
import logging


logger = logging.getLogger(__name__)


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


def test_parser_two_statements():
    """
    """
    text = ''' get_sample_data(end='2020') >> where("values_1 > 200")'''

    expected = """(get_sample_data(end='2020')\n.pipe(where, "values_1 > 200"))"""

    actual, call_function = pipe_parser(text, debug=False)

    assert expected == actual


# test_join_function_parms_single_statement {{{1
def test_join_function_parms_single_statement():

    statements = [('get_sample_sales(', ')')]

    expected = ['get_sample_sales()']
    actual = [join_function_parms(idx, x) for idx, x in enumerate(statements)]

    assert expected == actual


# test_join_function_parms_no_parms {{{1
def test_join_function_parms_no_parms():

    statements = [('get_sample_sales(', ')'),
                  ('select(', ')'),
                  ('head(', ')')]

    expected = ['get_sample_sales()',
                '.pipe(select)', '.pipe(head)']

    actual = [join_function_parms(idx, x) for idx, x in enumerate(statements)]

    assert expected == actual


# test_join_function_parms_single_parm {{{1
def test_join_function_parms_single_parm():

    statements = [('get_sample_sales(', ')'),
                  ('select(', ')'),
                  ('head(', 'n=4)')]

    expected = ['get_sample_sales()',
                '.pipe(select)', '.pipe(head, n=4)']

    actual = [join_function_parms(idx, x) for idx, x in enumerate(statements)]

    assert expected == actual


# test_join_function_parms_multi_parms {{{1
def test_join_function_parms_multi_parms():

    statements = [('get_sample_sales(', ')'),
                  ('select(', "['-date', 'something_else'])"),
                  ('head(', 'n=4)')]

    expected = ['get_sample_sales()',
                ".pipe(select, ['-date', 'something_else'])",
                '.pipe(head, n=4)']

    actual = [join_function_parms(idx, x) for idx, x in enumerate(statements)]

    assert expected == actual


