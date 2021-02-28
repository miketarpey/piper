![PyPI version shields.io](https://img.shields.io/pypi/v/dpiper)
![PyPI version shields.io](https://img.shields.io/pypi/pyversions/dpiper)
![PyPI version shields.io](https://img.shields.io/pypi/l/dpiper)

# Piper
__Piper__ is a python module to simplify data wrangling with [pandas](https://pandas.pydata.org/) using a set of wrapper functions. These functions or 'verbs' attempt to provide a simpler interface to standard Pandas functions.

When combined within a [Jupyter](https://jupyter.org/) notebook using the  __%%piper__ magic command, a simple data 'pipeline' that looks rather like SQL syntax can be built.

The concept is similar to the way R's [tidyverse](https://www.tidyverse.org/) and 
[magrittr](https://magrittr.tidyverse.org/) libraries are used. 

The main dataframe manipulation functions are:
- select()
- assign()
- relocate()
- where()
- group_by()
- summarise()
- order_by()

For other _piper_ functionality, please see the [Goals and Features](#Goals-and-Features) section.

___Alternatives___ 

For a comprehensive alternative, please check out __Michael Chow's [siuba package](https://github.com/machow/siuba)__. 

## Table of contents
* [Installation](#Installation)
* [Basic use](#Basic-use)
* [Documentation](#Documentation)
* [Goals and Features](#Goals-and-Features)
* [Status](#Status)
* [Inspiration](#Inspiration)
* [Contact](#Contact)

## Installation 
To install the package, enter the following:

```unix
pip install dpiper
```

## Basic use
Suppose you need the following function to trim a given dataframes columnar text data.

```python
def trim_columns(df):
    ''' Trim blanks for given dataframe '''
    
    str_cols = df.select_dtypes(include='object').columns
    
    for col in str_cols:
        df[col] = df[col].str.strip()
    
    return df
```

Standard [Pandas](https://pandas.pydata.org/) can combine the new function into a pipeline along with other transformation/filtering tasks:

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

### __Using piper__
Piper tries to improve this pipeline approach. Let's import piper's %%piper magic command and piper 'verbs'. 

```python
from piper import piper
from piper.verbs import head, select, where, group_by, summarise, order_by
```

Using the __%%piper__ magic function, piper verbs can be combined with standard python functions like trim_columns() using the linking symbol __'>>'__ to form a data pipeline.

```python
%%piper
get_sample_data()
>> trim_columns()
>> select('-regions')
>> where(""" ~countries.isin(['Italy', 'Portugal']) &
              values_1 > 40 &
              values_2 < 25 """)
>> order_by('countries', ascending=False)
>> head(5)
```

## Goals and Features

- Enhance working with Excel files through the WorkBook class 
    - Exporting high quality formatted Excel Workbooks using [xlsxwriter](https://xlsxwriter.readthedocs.io/)
- Provide access to databases with support for SQL based scripting and connections.

___To-do list___
* TBD 

## Documentation
Further examples are available in these jupyter notebooks:
- [piper demo jupyter notebooks](https://github.com/miketarpey/piper_demo)

## Status
Project has just started. I welcome any and all help to improve etc.

## Inspiration
Pandas and numpy are amazing data analysis libraries. My goal is to combine their power with the convenience and ease of use of the R tidyverse package suite.

## Contact
This is very much a personal library, in that its highly opinionated, flawed and probably of
no use to anyone else :). However if it helps anyone else in their endeavours, that would be fantastic to hear about.

If you'd like to contact me, I'm [miketarpey@gmx.net](mailto:miketarpey@gmx.net). 