![PyPI version shields.io](https://img.shields.io/pypi/v/dpiper)
![PyPI version shields.io](https://img.shields.io/pypi/pyversions/dpiper)
![PyPI version shields.io](https://img.shields.io/pypi/l/dpiper)

# Piper

__Piper__ is a python package to help simplify data wrangling tasks with
[pandas](https://pandas.pydata.org/). It provides a set of wrapper functions or
'verbs' that provide a simpler interface to standard Pandas functions.

Piper functions accept and receive pandas dataframe objects. They can be used as
standalone functions but are more powerful when used together in a
[Jupyter](https://jupyter.org/) notebook cell to form a data pipeline. This is
achieved by linking the functions using the __'>>'__ link operator within a cell
using __%%piper__ magic command.

Instead of the traditional pandas method of calling a method associated with
an object, in this case showing the 'head' (first 5 rows) of the dataframe:

```python
df.head()
```

Piper passes the result of the dataframe object as the first parameter to the
next function in the pipeline that, in turn, both accepts and returns dataframe
objects. So the equivalent of above with piper is:

```python
%%piper
df >> head()
```

This 'chaining' or linking of functions provides a rapid, easy to use/remember
approach to exploring, cleaning or building a data pipeline from csv, xml,
excel, databases etc. Custom functions that accept and return dataframe objects
can be linked together using this kind of syntax:

```python
%%piper
read_oracle_database()
>> validate_data()
>> cleanup_data()
>> generate_summary()
>> write_to_target_system()
```

## Table of contents
* [Installation](#Installation)
* [Basic use](#Basic-use)
* [Documentation](#Documentation)


## Installation
To install the package, enter the following:

```unix
pip install dpiper
```
<p>

## Basic use
__Example #1__ - A dataframe consisting of two columns A and B.

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

Let's create two further calculated columns and filter the 'D' column values.

```python
df['C'] = df['A'] + df['B']
df['D'] = df['C'] < 1000
df[df['D'] == False]
```
|    |   A |   B |    C | D     |
|---:|----:|----:|-----:|:------|
|  2 | 870 | 340 | 1210 | False |
|  8 | 624 | 673 | 1297 | False |


The equivalent in __piper__ would be:

```python
%%piper
df >> assign(C = lambda x: x.A + x.B,
             D = lambda x: x.C < 1000)
   >> where("~D")
```

__Example #2__ Suppose you need the following function to trim columnar text data.

```python
def trim_columns(df):
    ''' Trim blanks for given dataframe '''

    str_cols = df.select_dtypes(include='object').columns

    for col in str_cols:
        df[col] = df[col].str.strip()

    return df
```

Standard [Pandas](https://pandas.pydata.org/) can combine the new function into
a pipeline along with other transformation/filtering tasks by using the .pipe
method:

```python
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
```

Result:
| dates | order_dates | countries | ids | values_1 | values_2 |
| ----- | ----------- | --------- | --- | -------- | -------- |
2020-03-03 | 2020-03-09 | Sweden | E |	194  |20
2020-05-02 | 2020-05-08 | Sweden | D |	322  |14
2020-01-20 | 2020-01-26 | Spain  | A |  183  |20
2020-02-01 | 2020-02-07 | Norway | D |	344  |21
2020-05-06 | 2020-05-12 | Norway | B |	135  |21

The equivalent in __piper__ would be to import the piper magic function, and the
required 'verbs'.

```python
from piper import piper
from piper.verbs import head, select, where, group_by, summarise, order_by
```

Using the __%%piper__ magic function, piper verbs are combined with standard
python functions (e.g.) trim_columns() using the linking symbol __'>>'__ to form a
data pipeline.

```python
%%piper
get_sample_data()
>> trim_columns()
>> select('-dates')
>> where(""" ~countries.isin(['Italy', 'Portugal']) &
              values_1 > 40 &
              values_2 < 25 """)
>> order_by('countries', ascending=False)
>> head(5)
```

## Documentation
Piper API documentation available at [readthedocs](https://dpiper.readthedocs.io/en/latest/)
