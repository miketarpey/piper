Pre-requisite - Differences between Pandas method chaining vs piper
===================================================================

Pandas method chaining utilizes the DataFrame 'pipe' built-in method.

The pipe function takes:

  - the cleanup function required to be called followed by
  - additional arguments or keyword arguments required by the function.

Each invocation of the pipe method calls the function, passing the additional
parameters, then passing the result to the next pipe statement.

Expressions requiring more than one line are enclosed within parenthesis.
This 'dot' notation shown below is very powerful in that it encapsulates all the
filtering and transformational processes within a single pipeline.

.. code-block:: python

    (df
      .pipe(validate_data)
      .pipe(cleanup_data)
      .pipe(generate_summary, optional_parm='test')
      .pipe(write_to_target_system)
    )

Piper functions can use this dot notation above but also offers an alternative
syntax.

The **%%piper** magic command enables piper functions to be linked within a
Jupyter cell using the **'>>'** link operator.

Piper parses the text and creates the pipe expression in the background. It
takes the result of each function call and passes it to the next function as the
first parameter and so on until the end of the statement/expression.

The equivalent piper magic cell statement would be:

.. code-block:: python

    %%piper
    read_oracle_database()
    >> validate_data()
    >> cleanup_data()
    >> generate_summary(optional_parm='test')
    >> write_to_target_system()


The piper syntax:

  - does not require enclosing parenthesis

  - is more intuitive, the 'noise' of .pipe method usage is removed.

    To the user, it appears that the '>>' pipe operator is 'calling'
    the cleanup/transformation function normally.

The concept is based on the approach used in the R language
`tidyverse <https://www.tidyverse.org/>`_ and `magrittr
<https://magrittr.tidyverse.org/>`_ packages.

**Alternatives**

This is not a new idea and there are plenty of other Python solutions that
implement (in my opinion) far more elegant solutions.

For a near perfect port of R's tidyverse functionality, check out **Michael
Chow's** `siuba <https://github.com/machow/siuba>`_ package.







