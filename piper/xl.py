import logging
import re
import pandas as pd
from os.path import split, abspath, dirname, join
from IPython.display import display, HTML
from copy import deepcopy
from piper.configure import get_config
from piper.io import _get_qual_file
from piper.io import _file_with_ext
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name
from xlsxwriter.utility import xl_rowcol_to_cell

from pathlib import Path

logger = logging.getLogger(__name__)


# WorkBook() {{{1
class WorkBook():
    ''' WorkBook : Excel WorkBook helper class

    Encapsulates pandas DataFrames and XlsxWriter objects

    Parameters
    ----------
    file_name : (string) - the name of the Excel file to create.
                If you don't specify .xlsx extension, defaults to '.xlsx'

    sheets    : (df, list, dict) - multisheet mode

                1. A single DataFrame e.g. df
                file_name = 'outputs/single sheet.xlsx'
                WorkBook(file_name, df);

                2. A list of DataFrames e.g. [df1, df2, df3]
                file_name = 'outputs/multi sheet.xlsx'
                WorkBook(file_name, [df, df2]);

                3. A dictionary of sheet names and dataframe objects
                file_name = 'outputs/multi sheet with sheet names.xlsx'
                WorkBook(file_name, {'revised': df, 'original': df_original});

    ts_prefix : (string/boolean) - timestamp file name prefix.
                 - 'date' (date only) -> default
                 - False (no timestamp)
                 - True (timestamp prefix)

    date_format: (string) default 'dd-mmm-yy'

    datetime_format: (string) default 'dd-mmm-yy'


    Examples
    --------
    # Create an Excel workbook without a date/time prefix

    wb = WorkBook('outputs/excel workbook.xlsx', ts_prefix=False)
    wb.add_sheet(df)
    wb.close()

    # Create an Excel workbook, with two worksheets,
    # one with red tab sheet and zoom of 120%

    xl_file = f'excel workbook with date prefix.xlsx'
    wb = WorkBook(join('outputs', xl_file), ts_prefix='date')

    wb.add_sheet(df, sheet_name='revised data', tab_color='red', zoom=120)
    wb.add_sheet(df_original, sheet_name='original', tab_color='red')

    wb.close()

    # Create an Excel workbook, with worksheet containing
    # conditional formatting -> see ?WorkBook.add_condition for more details.

    %run ../src/defaults.py
    import pandas as pd

    data = {'Name':['Tom', 'nick', 'krish', 'jack'], 'Age':[20, 21, 19, 18]}
    df = pd.DataFrame(data)

    wb = WorkBook('outputs/excel workbook.xlsx', ts_prefix=False)

    sheet_name = 'test_worksheet'
    ws = wb.add_sheet(df, sheet_name=sheet_name, zoom=235)

    c={'type': 'formula', 'criteria': '=$B2<21', 'format': 'accent2', 'range': 'B'}
    wb.add_condition(ws, c)

    wb.close()
    '''
    sheet_dict = {}
    last_sheet_idx = 0

    def __init__(self, file_name, sheets=None, ts_prefix='date',
                 date_format='dd-mmm-yy', datetime_format='dd-mmm-yy'):
        ''' WorkBook constructor
        '''
        # Consider timestamp prefix parameter.
        folder, file = split(file_name)
        logger.debug(f'{folder}, {file}')
        file_name = _get_qual_file(folder, file, ts_prefix=ts_prefix)
        file_name = _file_with_ext(file_name)

        self.writer = pd.ExcelWriter(file_name, engine='xlsxwriter',
                                     datetime_format=datetime_format,
                                     date_format=date_format)
        self.wb = self.writer.book
        self.wb.set_properties(self.get_metadata())
        self.styles = self.get_styles()
        self.sheet_dict = {}  # Was getting memory issues !
        themes = WorkBook.get_themes()

        html = HTML(f"""<a href="{file_name}">{file_name}</a>""")
        display(html)
        logger.info(f'Workbook: {file_name}')

        if sheets is None:
            logger.info('<< sheet mode >>')
        else:
            logger.info('<< mult-sheet mode >>')
            self._save_xl(sheets)

    def _save_xl(self, sheets=None):
        ''' Save multiple Excel sheets in one WorkBook.
        Simplified Excel workbook object for quick creation & manipulation
        (Using the WorkBook object)
        '''

        # If sheet is just a dataframe, create a single sheet WorkBook
        if isinstance(sheets, pd.core.frame.DataFrame):
            self.add_sheet(sheets)

        # If sheet is a list of dataframe, create sheets for each one
        if isinstance(sheets, list):
            for df in sheets:
                self.add_sheet(df)

        # If sheet is a list of dataframe, create sheets for each one
        if isinstance(sheets, dict):
            for sheet, dataframe in sheets.items():
                self.add_sheet(dataframe, sheet_name=sheet)

        self.close()

    def add_sheet(self, dataframe=None, sheet_name='**auto', table=True,
                  header=True, index=False, startrow=0, startcol=0,
                  options=None, zoom=100, freeze_panes='A2', tab_color=None,
                  theme='Medium 2', sql=None):
        ''' Add DataFrame to WorkBook object

        Example
        -------
        file_name = 'outputs/test.xlsx'
        wb = WorkBook(file_name, ts_prefix=False)

        sheet_name = 'sheet_test1'
        ws = wb.add_sheet(df_concat, sheet_name)

        c = {'type': 'text', 'criteria': 'containing', 'value': 'A103',
             'format': 'accent2', 'range': 'A'}
        wb.add_condition(ws, c)

        ws = wb.add_sheet(df2, sheet_name='sheet2')

        wb.close()

        Parameters
        ----------
        dataframe         :            - Pandas dataframe reference.
        sheet_name        : '**auto'   - Auto-generate sheetx where x is next available index.
        table             : True       - Generate Excel Table format.
        header            : True       - DataFrame contains header row.
        index             : False      - When writing DataFrame in sheet, do not
                                         include index.
        startrow          : 0          - int, default 0 > Upper left cell row to dump
                                         data frame.
        startcol          : 0          - int, default 0 > Upper left cell column to
                                         dump data frame.
        options           : None       - Pass additional Excel format options through
                                         this argument.
        zoom              : 100        - Percentage sheet zoom
        freeze_panes      : 'A2'       - Freeze the first row using A1 notation
        tab_color         : None       - Tab colour (https://youtu.be/qz8uV2ooStk)
        theme             : 'Medium2'  - Use 'Medium2' Excel table theme as default
        '''
        if theme not in WorkBook.get_themes():
            raise NameError('**ERROR: Theme {theme} not valid')

        if sheet_name == '**auto':
            self.last_sheet_idx += 1
            sheet_name = f'sheet{self.last_sheet_idx}'

        dataframe.to_excel(self.writer, sheet_name, index=index,
                           startrow=startrow, startcol=startcol,
                           header=header, merge_cells=False)
        ws = self.writer.sheets[sheet_name]

        ws.freeze_panes(freeze_panes)

        if options is None:
            options = {'autofilter': True}

        ws.set_zoom(zoom)

        if tab_color is not None:
            ws.set_tab_color(tab_color)

        # Store worksheet metadata including:
        # < sheet name (as key) >
        # [0] dataframe index names (usually none)
        # [1] dataframe row and column information
        # [2] header (True/False)
        # [3] index (True/False) info
        # [4] ws reference
        # [5] tab color
        # [6] zoom
        self.sheet_dict.update({sheet_name: (dataframe.shape,
                                             dataframe.index.names,
                                             header, index, ws,
                                             tab_color, zoom)})

        sheet_range = self.get_range(sheet_name, startrow=startrow,
                                     startcol=startcol)
        logger.info(f'Sheet (range): {sheet_name} ({sheet_range})')

        if dataframe.columns.nlevels == 1:
            for ix, width in enumerate(WorkBook._calc_width(dataframe)):
                ws.set_column(ix, ix, width)

        if index is True:
            columns = dataframe.index.names + dataframe.columns.tolist()
        else:
            columns = dataframe.columns.tolist()

        if table is True:
            if dataframe.shape[0] >= 1:
                if header is True:
                    wrap_format = self.wb.add_format({'text_wrap': 1})
                    options.update({'columns': [{'header': x,
                                                 'header_format': wrap_format}
                                    for x in columns]})

                table_style = 'Table Style ' + theme
                options.update({'style': table_style, 'autofilter': True})
                ws.add_table(sheet_range, options)

        # TODO::
        # Include code below to allow user(s) to 'manually' write
        # excel worksheet data for greater control of formatting.
        #
        # # Write header
        # for col_num, value in enumerate(dataframe.columns.values):
        #     ws.write(0, col_num, value, self.styles.get('header'))

        # # Write detail
        # for row_no, row in enumerate(dataframe.iterrows()):
        #     offset = row_no + 1
        #     for col_no, col in enumerate(list(row)[1]):
        #         data_type = dataframe[dataframe.columns[col_no]].dtype
        #         if data_type == 'object' or str(data_type)[0:3] == 'int':
        #             style = 'sheet_default'
        #             ws.write(offset, col_no, col, self.styles.get(style))
        #         elif data_type == 'datetime64[ns]':
        #             style = 'date_dmy2'
        #             ws.write(offset, col_no, col, self.styles.get(style))
        #         else:
        #             style = 'price2'
        #             ws.write(offset, col_no, col, self.styles.get(style))

        # (Optionally) create sheet to store SQL
        if sql not in (None, ''):
            self.add_sql_sheet(sheet_name=sheet_name, sql=sql)

        return ws

    def add_sql_sheet(self, sheet_name=None, sql=None):
        ''' Add worksheet containing SQL statement(s) used '''

        sheet_name = '_' + sheet_name
        ws = self.wb.add_worksheet(sheet_name)
        ws.set_column('A:A', 120)
        ws.write('A1', sql, self.styles.get('sql'))
        ws.set_zoom(100)

        self.sheet_dict.update({sheet_name: ((1, 1), None, False, False, ws)})
        logger.info(f'Sheet: (SQL) {sheet_name}')

        return ws

    def get_range(self, sheet_name, column_range=None,
                  startrow=0, startcol=0, cond_override=False):
        ''' Given worksheet & column reference, calculate excel sheet range.
        e.g. get_column_range(df, 'D') returns $D$2:$D$12345

        Quick checks:
        -------------
        assert(get_range(sheet_name) == '$A$2:$R$3024')
        assert(get_range(sheet_name, 'D') == '$D$2:$D$3024')
        assert(get_range(sheet_name, 'D:M') == '$D$2:$M$3024')
        '''
        logger.debug(self.sheet_dict)
        df_shape = self.sheet_dict.get(sheet_name)[0]
        index_names = self.sheet_dict.get(sheet_name)[1]
        header = self.sheet_dict.get(sheet_name)[2]
        index = self.sheet_dict.get(sheet_name)[3]

        if index_names is None:
            index_len = 0
        else:
            index_len = len(index_names)

        total_cols = df_shape[1] - 1 + startcol
        if not index:
            columns = xl_col_to_name(total_cols, col_abs=True)
        else:
            columns = xl_col_to_name(index_len + total_cols, col_abs=True)

        logger.debug(columns)
        total_rows = df_shape[0] + startrow

        if header:
            row = 1 + startrow

            if cond_override:
                row = 2 + startrow

            rows = total_rows + 1
        else:
            row = 0 + startrow
            rows = total_rows

        if column_range is None or column_range == '':
            column = xl_col_to_name(0, col_abs=True)
            range_ref = f'{column}${row}:{columns}${rows}'
        else:
            column_range.upper()
            match = re.search(r'^(\w+):?(\w+)?', column_range, re.I)
            from_col, to_col = match[1], match[2]

            if to_col is None:  # e.g. 'D' -> '$D$2:$D$12345'
                range_ref = f'${column_range}${row}:${column_range}${rows}'
            else:  # e.g. 'D:M' -> '$D$2:$M$12345'
                range_ref = f'${from_col}${row}:${to_col}${rows}'

        logger.debug(f'Range ref: {range_ref}')
        return range_ref

    def add_format(self, worksheet=None, column_attr=None):
        ''' Add worksheet column format (rule)

        Parameters
        ----------
        worksheet   (Workbook.worksheet) worksheet object reference

        column_attr column attribute(s) to apply (dict or list)

                    You can pass a dictionary containing a single format change or,
                    pass a list of dictionary format definitions.

                    Pass the following 'keywords', 'column', 'width', 'format'

                    for list of formats or styles, type wb.get_styles()

        Examples
        --------
        # Set individual columnar format
        ws = wb.add_sheet(df_original, sheet_name='original', tab_color='yellow', zoom=175)

        wb.add_format(ws, column_attr={'column': 'A', 'width': 10, 'format': 'center_wrap'})
        wb.add_format(ws, column_attr={'column': 'B', 'width': 11})
        wb.add_format(ws, column_attr={'column': 'C', 'format': 'center', 'width': 25})

        # Pass a list of dictionary formats:
        # Below, example to quickly set widths for a range of columns
        cols = ['E', 'F', 'G', 'H', 'I']
        formats = [{'column': f'{c}', 'width': 10} for c in cols]

        wb.add_format(ws, column_attr=formats)

        Advanced usage
        --------------
        ## --START--
        wb = WorkBook(f'outputs/workbook.xlsx', ts_prefix='date')

        ws = wb.add_sheet(gblstprc, sheet_name='GBLSTPRC - SP, EQ', zoom=90,
                          tab_color='green', theme='Medium 16', freeze_panes='I2')
        format_sheet(wb, ws)

        wb.close()
        ## --END--

        def format_sheet(wb, ws):
            # Center the following columns
            centered_cols = {'A': 10, 'B': 10, 'C': 6, 'D': 20, 'F': 6,
                             'H': 7, 'J': 14, 'K': 7, 'L': 7, 'M': 6, 'O':9, 'R': 7}
            for col, width in centered_cols.items():
                wb.add_format(ws, {'column': col, 'width': width, 'format': 'center'})

            wb.add_format(ws, {'column': 'Q', 'format': 'price2'})

         def format_sheet2(wb, ws):

             centred_columns = [ {'column': 'A', 'width': 12}, {'column': 'B', 'width': 7},
                                 {'column': 'D', 'width': 11.29}, {'column': 'F', 'width': 9.43},
                                 {'column': 'G', 'width': 20}, {'column': 'H', 'width': 9.45},
                                 {'column': 'I', 'width': 9.45}, {'column': 'J', 'width': 9.14},
                                 {'column': 'L', 'width': 6.71} ]

             [d.update({'format': 'center_wrap'}) for d in centred_columns]

             other_formats = [{'column': 'C', 'width': 50}, {'column': 'E', 'width': 40},
                              {'column': 'K', 'width': 95}]

             wb.add_format(ws, column_attr=centred_columns + other_formats)
        '''
        if isinstance(column_attr, list):
            for attr in column_attr:
                self._set_format(worksheet, attr)

        elif isinstance(column_attr, dict):
            self._set_format(worksheet, column_attr)

    @staticmethod
    def get_default_format(df):
        ''' For given dataframe, calculate default format.

        This function provides greater control of formatting of individual
        columns.

        Example:
        ========
        df = get_sample_data()
        formats = get_default_format(df)

        >> [{'column_name': 'dates', 'column': 'A', 'width': 11},
        >>  {'column_name': 'order_dates', 'column': 'B', 'width': 12},
        >>  {'column_name': 'countries', 'column': 'C', 'width': 12},
        >>  {'column_name': 'regions', 'column': 'D', 'width': 8},
        >>  {'column_name': 'ids', 'column': 'E', 'width': 4},
        >>  {'column_name': 'values_1', 'column': 'F', 'width': 9},
        >>  {'column_name': 'values_2', 'column': 'G', 'width': 9}]

        ---
        wb = WorkBook('outputs/testing output format')

        ws = wb.add_sheet(df, sheet_name='testing')

        [x.update({'format': 'center_wrap'}) for x in formats]

        wb.add_format(ws, column_attr=formats)

        wb.close()

        '''
        columns = df.columns.tolist()

        f = lambda x, y: {'column_name': y, 'column': xl_rowcol_to_cell(0, x)[0]}
        default_setup = [f(idx, col) for idx, col in enumerate(columns)]

        # Add default widths
        widths = WorkBook._calc_width(df).astype(int)

        for idx, dict_ in enumerate(default_setup):
            dict_.update({'width': widths[idx]})

        return default_setup

    def _set_format(self, worksheet, column_attr):
        ''' Set column format attributes for specified worksheet

        Parameters
        ----------
        worksheet   (Workbook.worksheet) worksheet object reference

        column_attr column attribute(s) to apply (dict or list)

                    You can pass a dictionary containing a single condition or,
                    pass a list of dictionary format definitions.

                    Pass the following 'keywords', 'column', 'width', 'format'

                    for list of formats or styles, type wb.get_styles()
        '''
        column = column_attr.get('column').upper()
        column_str = f'{column}:{column}'
        width = column_attr.get('width')
        format_style = column_attr.get('format')

        if format_style is not None:
            style = self.styles.get(format_style)

        if width is not None and format_style is None:
            worksheet.set_column(column_str, width=width, cell_format=None)

        # NOTE:: when passing NONE width, defaults to default column width
        if width is None and format_style is not None:
            worksheet.set_column(column_str, width=width, cell_format=style)

        if width is not None and format_style is not None:
            worksheet.set_column(column_str, width=width, cell_format=style)

    def add_condition(self, worksheet=None, condition=None):
        ''' Add worksheet conditional format (rule)

        Parameters
        ----------
        worksheet   (Workbook.worksheet) worksheet object reference

        condition   condition(s) information to apply (dict or list)

                    You can pass a dictionary containing a single condition or,
                    pass a list of dictionary definitions.

                    NOTE: if 'format' not specified, then 'error' format is
                    defaulted.
                    if 'range' not specified, then condition applies to
                    entire row.

                    Valid condition types are:
                    ('2_color_scale', '3_color_scale', 'type', 'average', 'blanks',
                     'bottom', 'cell', 'data_bar', 'date', 'duplicate', 'errors',
                     'formula', 'icon_set', 'no_blanks', 'no_errors', 'text',
                     'time_period', 'top', 'unique')

        For more details, consult link below.
        # https://xlsxwriter.readthedocs.io/working_with_conditional_formats.html

        Examples
        --------
        xl_file = f'example #7 WorkBook - Multi sheet with conditional formatting.xlsx'
        wb = WorkBook(f'outputs/{xl_file}'), ts_prefix=None)

        ws = wb.add_sheet(df_original, sheet_name='original', tab_color='yellow', zoom=175)

        ws = wb.add_sheet(df, sheet_name='revised data', tab_color='red', zoom=175)

        # Simple (single dictionary) condition
        ws = wb.add_condition(ws, condition={'type': '3_color_scale', 'range': 'H'})

        selected_date = datetime.strptime('2018-01-01', "%Y-%m-%d")

        # Multiple conditions added to a sheet (list of dictionaries)
        c = [ {'type': 'formula', 'criteria': '=$I2=2263', 'format': 'accent4'},
              {'type': 'cell', 'criteria': 'equal to', 'value': '"A103"', 'format': 'accent5', 'range': 'A'},
              {'type': 'cell', 'criteria': 'equal to', 'value': 23899003, 'format': 'accent6', 'range': 'B'} ,
              {'type': 'duplicate', 'format': 'accent1', 'range': 'C:D'},
              {'type': 'text', 'criteria': 'containing', 'value': 'Eoin', 'format': 'accent2', 'range': 'D'},
              {'type': 'data_bar', 'data_bar_2010': True, 'criteria': '=$F2>0', 'range': 'F'},
              {'type': 'date', 'criteria': 'less than', 'value': selected_date, 'format': 'accent3', 'range': 'G'},
              {'type': 'formula', 'criteria': '=$J2="cat;dog;books"', 'format': 'accent5', 'range': 'J'},
           ]

        wb.add_condition(ws, condition=c)

        c = [ {'type': 'formula', 'criteria': '=$B2=""',   'format': 'accent2', 'range': 'B'},
              {'type': 'formula', 'criteria': '=$M2=""',   'format': 'accent2', 'range': 'M'},
              {'type': 'formula', 'criteria': '=$F2="O"',  'format': 'input'},
              {'type': 'formula', 'criteria': '=$J2="uom mismatch"', 'format': 'accent1', 'range': 'J:L'}
           ]

        wb.add_condition(ws, c)

        wb.close()
        '''
        def get_attribute(worksheet, c, name=None):

            try:
                attribute_value = c.pop(name)
            except KeyError as e:
                attribute_value = None

            column_range = self.get_range(worksheet.get_name(),
                                          column_range=attribute_value,
                                          cond_override=True)

            return column_range

        default_format = {'format': self.styles.get('error')}

        logger.debug(condition)

        if isinstance(condition, list):

            for c in condition:

                format_style = c.get('format')
                if format_style is None:
                    c.update(default_format)
                else:
                    c.update({'format': self.styles.get(format_style)})

                column_range = get_attribute(worksheet, c, 'range')
                worksheet.conditional_format(column_range, c)

        elif isinstance(condition, dict):

            format_style = condition.get('format')
            if format_style is None:
                condition.update(default_format)
            else:
                condition.update({'format': self.styles.get(format_style)})

            column_range = get_attribute(worksheet, condition, 'range')
            worksheet.conditional_format(column_range, condition)

    def get_metadata(self, meta_file=None):
        ''' Get XL metadata '''

        if meta_file is None:
            config = get_config('config.json', info=False)
        else:
            xl_meta = get_config(meta_file, info=False)

        xl_meta = get_config(config['excel']['meta'], info=False)

        return xl_meta

    @staticmethod
    def _calc_width(df):
        ''' Given dataframe, calculate optimum column widths

        Example:
        for ix, width in enumerate(_calc_width(df)['max_']):
            ws.set_column(ix, ix, width)
        '''
        maxlen_colnames = [len(x) for x in df.columns]
        max_col_names = pd.DataFrame(maxlen_colnames, columns=['max_col_names'])

        maxlen_cols = [max(df.iloc[:, col].astype(str).apply(len))
                       for col in range(len(df.columns))]
        max_cols = pd.DataFrame(maxlen_cols, columns=['max_cols'])

        widths = pd.concat([max_col_names, max_cols], axis=1, sort=False)

        # http://polymathprogrammer.com/2010/01/18/calculating-column-widths-in-excel-open-xml/
        f = lambda x: x + x*((256/x)/256)
        widths['max_'] = widths.max(axis=1).apply(f)

        return widths['max_']

    @staticmethod
    def show_themes():
        ''' Show/List themes '''
        return ', '.join(WorkBook.get_themes())

    @staticmethod
    def get_themes():
        ''' Get MS Excel standard themes '''

        light_list = [f'Light {x}' for x in list(range(1, 29))]
        medium_list = [f'Medium {x}' for x in list(range(1, 29))]
        dark_list = [f'Dark {x}' for x in list(range(1, 13))]

        lists = [light_list, medium_list, dark_list]
        themes = [item for sublist in lists for item in sublist]

        return themes

    def show_styles(self):
        ''' Show/List internal styles '''
        return self.styles.keys()

    def get_styles(self, file_name=None):
        ''' Retrieve dictionary of styles allowing easy access
        to styles by name.
        '''
        if file_name is None:
            config = get_config('config.json')
        else:
            config = get_config(file_name)

        styles_dict = get_config(config['excel']['formats'])

        base_fmt = {'font_name': 'Calibri', 'font_size': 11}

        def set_format(v):
            v.update(deepcopy(base_fmt))
            return self.wb.add_format(v)

        styles = {k: set_format(v) for k, v in styles_dict.items()}
        default = {'sheet_default': set_format(base_fmt)}
        styles = {**styles, **default}

        return styles

    def close(self):
        ''' close (save) workbook '''
        try:
            self.writer.save()
            # self.wb.close()
        except PermissionError as e:
            logger.info(e)
        except xlsxwriter.exceptions.FileCreateError as e:
            logger.info(e)

        logger.info('Completed.')


# sqldeveloper_xl() {{{1
def sqldeveloper_xl(filename=None, target=None, ts_prefix='date',):

    if filename is None:
        project_dir = Path.home()
        filename = project_dir / 'export.xlsx'

    if target is None:
        project_dir = Path.home()
        target = project_dir / 'documents/export.xlsx'

    data = pd.read_excel(filename)

    sql_stmt = pd.read_excel(filename, sheet_name='SQL')
    sql = sql_stmt.iloc[:1, 0].name

    wb = WorkBook(target, ts_prefix=ts_prefix)
    wb.add_sheet(data, sql=sql)
    wb.close()
