import pytest
from piper.jde import add_jde_batch
from piper.test.factory import get_sample_df1


# sample_df1 {{{1
@pytest.fixture
def sample_df1():
    return get_sample_df1()


# test_add_jde_batch {{{1
def test_add_jde_batch(sample_df1):
    """
    """
    df = add_jde_batch(sample_df1)

    expected = (367, 11)
    actual = df.shape

    assert expected == actual
