Basic use
=========

Import piper within your Jupyter notebook. This will import the main *verbs*
and register the **%%piper** magic command to be available.


.. code-block::

    from piper.defaults import *


Example #1
----------

.. code-block:: python

    import pandas as pd
    from piper.factory import sample_sales

    df = sample_sales()

    # Subset to London data
    london_sales_beachwear = (df[(df['location'] == 'London') & 
                                 (df['product'] == 'Beachwear')])

    # Group the data and find the total sales of beachwear for each month.
    grouped_data = (london_sales_beachwear
                    .groupby(['location', 'product', 'month'])
                    .agg(TotalSales=('target_sales', 'sum')))

    grouped_data.head(4)

.. code-block::

    location    product    month       TotalSales
    ==========  =========  ==========  ==========
    London      Beachwear  2021-01-01       69582
    London      Beachwear  2021-03-01       28837
    London      Beachwear  2021-04-01       36395
    London      Beachwear  2021-05-01       69252

*Piper alternative*

.. code-block::

    from piper.defaults import *

.. code-block:: python

    %%piper

    sample_sales()
    >> where("location == 'London' & product == 'Beachwear'")
    >> group_by(['location', 'product', 'month'])
    >> summarise(TotalSales=('target_sales', 'sum'))
    >> head(4)


Example #2
----------

A dataframe consisting of two columns A and B.

.. code-block::

    import pandas as pd
    import numpy as np

    np.random.seed(42)

    df = pd.DataFrame({'A': np.random.randint(10, 1000, 10),
                       'B': np.random.randint(10, 1000, 10)})
    df.head()

    ..    A    B
    ==  ===  ===
     0  112  476
     1  445  224
     2  870  340
     3  280  468
     4  116   97

Let's create two further calculated columns and filter the 'D' column values.

.. code-block::

    df['C'] = df['A'] + df['B']
    df['D'] = df['C'] < 1000
    df[df['D'] == False]

      ..    A    B     C      D
    ====  ===  ===  ====  =====
       2  870  340  1210  False
       8  624  673  1297  False


The equivalent in __piper__ would be:

.. code-block::

    %%piper
    df
    >> assign(C = lambda x: x.A + x.B,
              D = lambda x: x.C < 1000)
    >> where("~D")

Example #3
----------

Suppose you need the following function to trim columnar text data.

.. code-block::

    def trim_columns(df):
        ''' Trim blanks for given dataframe '''

        str_cols = df.select_dtypes(include='object').columns

        for col in str_cols:
            df[col] = df[col].str.strip()

        return df

Standard `Pandas <https://pandas.pydata.org/>`_ can combine the new function
into a pipeline along with other transformation/filtering tasks by using the
.pipe method:

.. code-block::

    import pandas as pd
    from piper.factory import get_sample_data

    df = get_sample_data()

    # Select all columns EXCEPT 'dates'
    subset_cols = ['order_dates', 'regions', 'countries', 'values_1', 'values_2']

    criteria1 = ~df['countries'].isin(['Italy', 'Portugal'])
    criteria2 = df['values_1'] > 40
    criteria3 = df['values_2'] < 25

    df2 = (df[subset_cols][criteria1 & criteria2 & criteria3]
        .pipe(trim_columns)
        .sort_values('countries', ascending=False))

    df2.head()


The equivalent in __piper__ would be to import the piper magic function, and
the required 'verbs'.

.. code-block::

    from piper import piper
    from piper.verbs import *

Using the __%%piper__ magic function, piper verbs can be combined with standard python functions like str_trim() using the linking symbol __'>>'__ to form a data pipeline.

.. code-block::

    %%piper
    get_sample_data()
    >> str_trim()
    >> select('-dates')
    >> where(""" ~countries.isin(['Italy', 'Portugal']) &
                values_1 > 40 &
                values_2 < 25 """)
    >> order_by('countries', ascending=False)
    >> head(5)

**--info** option
If you specify this option, you see the equivalent pandas 'piped'
version below the cell.

.. code-block::

    %%piper --info
    get_sample_data()
    >> trim_columns()
    >> select('-dates')
    >> where(""" ~countries.isin(['Italy', 'Portugal']) &
                values_1 > 40 &
                values_2 < 25 """)
    >> order_by('countries', ascending=False)
    >> head(5)

gives:

.. code-block::

    (get_sample_data()
    .pipe(select, '-dates')
    .pipe(where, """ ~countries.isin(['Italy', 'Portugal']) &values_1 > 40 &values_2 < 25 """)
    .pipe(order_by, 'countries', ascending=False)
    .pipe(head, 5))
