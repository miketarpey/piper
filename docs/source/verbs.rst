verbs module
============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. currentmodule:: piper.verbs

selecting / filtering / ordering
--------------------------------

.. autosummary::
   :toctree: api

   select
   where
   order_by

data analysis / review
----------------------

.. autosummary::
   :toctree: api

   count
   head
   info
   memory
   sample
   tail

column management
-----------------

.. autosummary::
   :toctree: api

   columns
   flatten_cols
   relocate
   rename
   rows_to_columns
   set_columns

data cleaning
-------------

.. autosummary::
   :toctree: api

   clean_columns
   distinct
   drop_columns
   drop
   duplicated
   overlaps
   non_alpha

assign/update column(s)
-----------------------

.. autosummary::
   :toctree: api

   across
   assign

string functions
----------------

.. autosummary::
   :toctree: api

   str_join
   str_split
   str_trim

joining data
------------

.. autosummary::
   :toctree: api

   inner_join
   left_join
   right_join
   outer_join

aggregation
-----------

.. autosummary::
   :toctree: api

   adorn
   group_by
   summarise
   transform

reshaping data
--------------

.. autosummary::
   :toctree: api

   explode
   pivot_longer
   pivot_table
   split_dataframe
   stack
   summary_df
   unstack

index management
----------------

.. autosummary::
   :toctree: api

   fmt_dateidx
   rename_axis
   reset_index
   set_index
