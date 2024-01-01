import sqlite3 as sql
from os.path import join
from typing import Dict
import re
from nltk.util import everygrams
import os

import pandas as pd

"""
Tools to access the databases in CoSQL.
"""


def PUNCTUATION_STRIPPER(s):
    '''Strips all punctuation from string s.'''
    s = re.sub('\s\s+', ' ', re.sub(r'[^\w\s\.]', '', s))
    # remove periods for words
    matches = re.findall('[a-z]+\.', s)
    for m in matches:
        s = s.replace(m, m[:-1])
    return s


def CLEAN_QUERY_RESULT_ROW(row):
    '''cleans a row from query execution.'''
    row = row[0]
    if row is None:
        return '__NONE__'
    row = str(row)
    if len(row) == 0:
        return '__EMPTY__'
    return PUNCTUATION_STRIPPER(row).lower().strip()

def CLEAN_DATAFRAME_ROW(row):
    '''cleans a row from query execution.'''
    if row is None:
        return '__NONE__'
    row = str(row)
    if len(row) == 0:
        return '__EMPTY__'
    return PUNCTUATION_STRIPPER(row).lower().strip()


class Database:
    '''
    Class to represent one SQL database in CoSQL. Uses SQLite.
    '''
    NUMERICAL_FLAG = 'Numerical'
    TEXT_FLAG = 'Textual'

    def __init__(self, id: str, parent_directory: str, schema: Dict, grams: int = -1):
        self.id = id
        self.parent_directory = parent_directory
        self.conn = None
        self.cur = None
        self.schema = schema
        # This is just to cache results so we don't have to keep accessing the db
        # Running the DB is a long operation in comparison
        self.query_results = None
        self.num_max_grams = None
        if grams == -1:
            self.num_max_grams = 5
        else:
            self.num_max_grams = grams

    def _connect(self):
        self.conn = sql.connect(
            join(os.getcwd(), self.parent_directory, self.id, self.id + '.sqlite'))
        self.cur = self.conn.cursor()

    def _close(self):
        self.conn.close()
        self.conn = None
        self.cur = None

    def _pull_column_data(self, column):
        '''
        Pulls all of the values from a particular column in the db.
        '''
        if self.query_results is not None:
            return
        self._connect()
        # First find the relevant column name
        human_columns = self.schema['column_names']
        # exact column names
        human_columns = [i[1] for i in human_columns]
        matches = []
        for count, _col in enumerate(human_columns):
            if _col == column:
                matches.append(count)
        if len(matches) == 0:
            assert False, f"ERROR: Could not find {column} in id {self.id}"
        self.query_results = []
        for idx in matches:
            # Now grab the original name of it (sql name)
            db_column_info = self.schema['column_names_original'][idx]
            db_column_name = db_column_info[1]
            numerical_query = self.schema['column_types'][idx] == 'number'
            if numerical_query:
                self.query_results.append(Database.NUMERICAL_FLAG)
            else:
                # Determine which table it belongs to
                table_index = db_column_info[0]
                table_name = self.schema['table_names_original'][table_index]

                # Now query the table
                sql_query = f"SELECT {db_column_name} from {table_name}"
                res = self.cur.execute(sql_query).fetchall()
                # print(res, db_column_name, self.schema['column_types'][idx])
                self.query_results.append(
                    [CLEAN_QUERY_RESULT_ROW(row) for row in res])
        self._close()

    def _numerical_check(self, ngram):
        '''tests if ngram is a number'''
        try:
            _t = float(ngram)
            return True
        except ValueError:
            return False

    def _text_check(self, ngram, column_values):
        '''tests if ngram is a valid string'''
        if len(ngram) == 0:
            return False
        return ngram in column_values

    def search_column(self, term, column):
        '''Searching for a term in the values of DB.column. Returns the ngrams which match and whether it is numerical or textual (see class flags).'''
        self._pull_column_data(column)
        term_tokens = PUNCTUATION_STRIPPER(term.strip().lower()).split(' ')
        term_ngrams = everygrams(term_tokens, max_len=self.num_max_grams)
        matches = []
        numerical_or_text = []
        for ngram in term_ngrams:
            ngram = ' '.join(ngram)
            for col_values in self.query_results:
                if col_values == Database.NUMERICAL_FLAG and self._numerical_check(ngram):
                    matches.append(float(ngram))
                    numerical_or_text.append(Database.NUMERICAL_FLAG)
                elif col_values != Database.NUMERICAL_FLAG and self._text_check(ngram, col_values):
                    if ngram.isnumeric():
                        matches.append(float(ngram))
                        numerical_or_text.append(Database.NUMERICAL_FLAG)
                    else:
                        matches.append(ngram)
                        numerical_or_text.append(Database.TEXT_FLAG)
        self.query_results = None
        return matches, numerical_or_text
    
class Dataframe:
    '''
    '''
    NUMERICAL_FLAG = 'Numerical'
    TEXT_FLAG = 'Textual'

    def __init__(self, df: pd.DataFrame, grams: int = -1):
        
        self.df = df

        self.query_results = None
        self.num_max_grams = None
        if grams == -1:
            self.num_max_grams = 5
        else:
            self.num_max_grams = grams

    def _pull_column_data(self, column):
        '''
        Pulls all of the values from a particular column in the db.
        '''
        if self.query_results is not None:
            return

        matches = []
        for count, _col in enumerate(self.df.columns):
            if _col == column:
                matches.append(count)
        if len(matches) == 0:
            assert False, f"ERROR: Could not find {column} in the dataframe"
        self.query_results = []
        for idx in matches:
            # Now grab the original name of it (sql name)
            db_column_name = self.df.columns[idx]
            # db_column_name = db_column_info[1]
            numerical_query = ("float" in str(self.df.dtypes[idx])) or ("int" in str(self.df.dtypes[idx]))
            if numerical_query:
                self.query_results.append(Dataframe.NUMERICAL_FLAG)
            else:
                res = self.df[self.df.columns[idx]]

                self.query_results.append(
                    [CLEAN_DATAFRAME_ROW(row) for row in res])

    def _numerical_check(self, ngram):
        '''tests if ngram is a number'''
        try:
            _t = float(ngram)
            return True
        except ValueError:
            return False

    def _text_check(self, ngram, column_values):
        '''tests if ngram is a valid string'''
        if len(ngram) == 0:
            return False
        return ngram in column_values

    def search_column(self, term, column):
        '''Searching for a term in the values of DB.column. Returns the ngrams which match and whether it is numerical or textual (see class flags).'''
        self._pull_column_data(column)
        term_tokens = PUNCTUATION_STRIPPER(term.strip().lower()).split(' ')
        term_ngrams = everygrams(term_tokens, max_len=self.num_max_grams)
        matches = []
        numerical_or_text = []
        for ngram in term_ngrams:
            ngram = ' '.join(ngram)
            for col_values in self.query_results:
                if col_values == Database.NUMERICAL_FLAG and self._numerical_check(ngram):
                    matches.append(float(ngram))
                    numerical_or_text.append(Database.NUMERICAL_FLAG)
                elif col_values != Database.NUMERICAL_FLAG and self._text_check(ngram, col_values):
                    if ngram.isnumeric():
                        matches.append(float(ngram))
                        numerical_or_text.append(Database.NUMERICAL_FLAG)
                    else:
                        matches.append(ngram)
                        numerical_or_text.append(Database.TEXT_FLAG)
        self.query_results = None
        return matches, numerical_or_text
