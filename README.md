![PyPI version shields.io](https://img.shields.io/pypi/v/dpiper)
![PyPI version shields.io](https://img.shields.io/pypi/pyversions/dpiper)
![PyPI version shields.io](https://img.shields.io/pypi/l/dpiper)

# Piper

__Piper__ is a python package to help simplify data wrangling tasks with
[pandas](https://pandas.pydata.org/). It provides a set of wrapper functions or
'verbs' that provide a simpler interface to standard Pandas functions.

Piper functions accept and receive pandas dataframe objects. They can be used as
standalone functions but are more powerful when used together in a
[Jupyter](https://jupyter.org/) notebook cell to form a data pipeline.

Instead of the traditional the 'dot notation' or method calling technique from
within an object, piper receives and passes on the dataframe object between
functions using the piper magic command link operator (that is __'>>'__) within
a cell. So, in traditional pandas, to see the first 5 rows of a dataframe:

```python
df.head()
```

The equivalent in piper would be:

```python
%%piper
df >> head()
```

# Installation
To install the package, enter the following:

```unix
pip install dpiper
```

# Documentation
Piper API documentation available at [readthedocs](https://dpiper.readthedocs.io/en/latest/)

# Quick start

## __Example #1__

A dataframe consisting of two columns A and B.

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({'A': np.random.randint(10, 1000, 10),
                   'B': np.random.randint(10, 1000, 10)})
df.head()
```

|    |   A |   B |
|---:|----:|----:|
|  0 | 112 | 476 |
|  1 | 445 | 224 |
|  2 | 870 | 340 |
|  3 | 280 | 468 |
|  4 | 116 |  97 |

### __Piper equivalent__

```python
from piper.defaults import *
```
```python
%%piper
df >> assign(C = lambda x: x.A + x.B,
             D = lambda x: x.C < 1000)
   >> where("~D")
```

|    |   A |   B |    C | D     |
|---:|----:|----:|-----:|:------|
|  2 | 870 | 340 | 1210 | False |
|  8 | 624 | 673 | 1297 | False |

## __Example #2__
Suppose you need the following function to trim columnar text data.

```python
def trim_columns(df):
    ''' Trim blanks for given dataframe '''

    str_cols = df.select_dtypes(include='object').columns

    for col in str_cols:
        df[col] = df[col].str.strip()

    return df

import pandas as pd
from piper.factory import sample_data

df = sample_data()

# Select all columns EXCEPT 'dates'
subset_cols = ['order_dates', 'regions', 'countries', 'values_1', 'values_2']

criteria1 = ~df['countries'].isin(['Italy', 'Portugal'])
criteria2 = df['values_1'] > 40
criteria3 = df['values_2'] < 25

df2 = (df[subset_cols][criteria1 & criteria2 & criteria3]
       .pipe(trim_columns)
       .sort_values('countries', ascending=False))
df2.head()
```

### __Piper equivalent__

Using the __%%piper__ magic function, piper verbs can be combined with standard
python functions.

```python
from piper.defaults import *
```

```python
%%piper
sample_data()
>> trim_columns()
>> select('-dates')
>> where(""" ~countries.isin(['Italy', 'Portugal']) &
              values_1 > 40 &
              values_2 < 25 """)
>> order_by('-countries')
>> head(5)
```

Result:
| dates | order_dates | countries | ids | values_1 | values_2 |
| ----- | ----------- | --------- | --- | -------- | -------- |
2020-03-03 | 2020-03-09 | Sweden | E |	194  |20
2020-05-02 | 2020-05-08 | Sweden | D |	322  |14
2020-01-20 | 2020-01-26 | Spain  | A |  183  |20
2020-02-01 | 2020-02-07 | Norway | D |	344  |21
2020-05-06 | 2020-05-12 | Norway | B |	135  |21
