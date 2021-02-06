# Piper
__Piper__ is a python module to simplify data wrangling with [pandas](https://pandas.pydata.org/).

Combined with a [Jupyter](https://jupyter.org/) notebook a 'magic' command (__%%piper__), provides an SQL like syntax - similar to R's [tidyverse](https://www.tidyverse.org/) and 
[magrittr](https://magrittr.tidyverse.org/) libraries

The main functions are:

- select()
- where()
- group_by()
- summarise()
- order_by()

___Alternatives___ 

For a comprehensive alternative, please check out __Michael Chow's [siuba package](https://github.com/machow/siuba)__. 

For other _piper_ functionality, please see the [Features](#Features) section.

## Table of contents
* [Installation](#Installation)
* [Basic use](#Basic-use)
* [Documentation](#Documentation)
* [Features](#Features)
* [Status](#Status)
* [Inspiration](#Inspiration)
* [Contact](#Contact)

## Installation 
To install the package, enter the following:

```
pip install dpiper
```

## Basic use
Within a Jupyter notebook, suppose we need a function to returned trimmed string based column data for a dataframe.
```python
def trim_columns(df):
    ''' Trim blanks for given dataframe '''
    
    str_cols = df.select_dtypes(include='object').columns
    
    for col in str_cols:
        df[col] = df[col].str.strip()
    
    return df
```

In standard pandas, we can combine the new function in a pipeline, along with filtering the input data as follows:

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

Now, the same result using piper's %%piper magic command and using piper 'verbs'. Let's import the necessary functions: 

```python
from piper import piper
from piper.verbs import head, select, where, group_by, summarise, order_by
```

Now, create a separate cell with functions 'piped' together using the linking symbol '>>'

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

## Features
- Simplifies working with data pipelines by implementing wrapper functions.
- Wrappers for working with Excel files (using [xlsxwriter](https://xlsxwriter.readthedocs.io/))
- Provides easy access to sql and database connections.

To-do list:
* TBD 

## Documentation
Further examples are available in these jupyter notebooks:
- TBD
- TBD
- TBD

## Status
Project has just started. I welcome any and all help to improve etc.

## Inspiration
Pandas, numpy are amazing libraries. I truly am standing on the shoulders of
giants.
Had I not been exposed to the R language and tidyverse - I would never have
bothered to even try building a package.

This is very much a personal library, in that its highly opinionated, flawed and probably of
no use to anyone else :). That said, if it helps you in your endeavours - fantastic!

## Contact
[miketarpey@gmx.net](mailto:miketarpey@gmx.net). 