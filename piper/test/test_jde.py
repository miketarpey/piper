from piper.jde import add_jde_batch
from piper.test.factory import get_sample_df1


# test_add_jde_batch {{{1
def test_add_jde_batch(get_sample_df1):
    """
    """
    dx = get_sample_df1
    df = add_jde_batch(dx, inplace=False)

    expected = (367, 11)
    actual = df.shape

    assert expected == actual
