import sqlite3
from mo_sql_parsing.utils import binary_ops
import numbers
from os.path import join
import itertools
import numbers
from src.common.utils import sql_state_tracking_json, \
    table_metadata_json
from src.config import cfg

"""
Set of functions for helping with manipulating CoSQL.
"""

# Binary operations
b_ops = ['<', '<=', '>', '>=', '=', '<=>', '!=',
         'like', 'rlike', 'not like', 'not rlike']
BINARY_OPS = [binary_ops[op] for op in b_ops]
arithmetic_ops = ['*', '/', '%', '+', '-']
ARITHMETIC_OPS = [binary_ops[op] for op in arithmetic_ops]


class Dialogue:
    '''
    Class representing a single CoSQL dialogue.
    '''

    def __init__(self, db_id):
        self.db_id = db_id
        self.interactions = []

    def add_interaction(self, utterance, query):
        self.interactions.append([utterance, query])

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        return self.interactions[idx]


def flatten_corpus(corpus):
    '''
    Given the entire set of CoSQL dialogues, flattens into one large interactions list.
    '''
    interactions = []
    for dialogue in corpus:
        for i in dialogue.interactions:
            interactions.append(i)
    return interactions


def build_corpus(split):
    '''
    Generates the dialogues from CoSQL.
    '''
    raw_json = sql_state_tracking_json(split)
    corpus = []
    for d in raw_json:
        current_dialogue = Dialogue(d['database_id'])
        for interaction in d['interaction']:
            current_dialogue.add_interaction(
                interaction['utterance'], interaction['query'])
        corpus.append(current_dialogue)
    return corpus


def build_database_metadata():
    '''
    Builds the metadata for CoSQL DB tables.
    '''
    raw_json = table_metadata_json()
    final_dbs = dict()
    for entity in raw_json:
        k = entity['db_id']
        raw_cols = entity['column_names_original']
        raw_cols = [i[1] for i in raw_cols if len(i) > 1 and i[1] != '*']
        cols = entity['column_names']
        cols = [i[1] for i in cols if len(i) > 1 and i[1] != '*']
        final_dbs[k] = {'original': [i.lower() for i in raw_cols],
                        'transformed': [i for i in cols]}
    return final_dbs

def get_value_set(db_id, col):
    '''
    Given a DB ID and column name, then list all possible values in the column.
    '''
    fname = join(cfg.DATA_TABLES, db_id, f'{db_id}.sqlite')
    conn = sqlite3.connect(fname)
    cursor = conn.cursor()
    # get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    res = cursor.fetchall()
    table_names = map(lambda x: x[0], res)

    # Now iterate through each column name to see if it matches
    values = set()
    for table_name in table_names:
        cursor.execute(f"SELECT * FROM {table_name}")
        col_names = list(map(lambda x: x[0], cursor.description))
        cursor.fetchall()
        if col in col_names:
            res = cursor.execute(f"SELECT {col} from {table_name}").fetchall()
            res = list(map(lambda x: x[0], res))
            values = values.union(set(res))
    return list(values)


def get_all_value_sets(db_id):
    '''
    Get the unique values for each column given a DB id.
    '''
    fname = join(cfg.DATA_TABLES, db_id, f'{db_id}.sqlite')
    conn = sqlite3.connect(fname)
    cursor = conn.cursor()
    # get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    res = cursor.fetchall()
    table_names = map(lambda x: x[0], res)
    values = dict()
    for table_name in table_names:
        cursor.execute(f"SELECT * FROM {table_name}")
        col_names = list(map(lambda x: x[0], cursor.description))
        cursor.fetchall()
        # print(col_names)
        for col in col_names:
            if col not in values.keys():
                values[col] = set()
            res = cursor.execute(
                f"SELECT '{col}' from {table_name}").fetchall()
            res = list(map(lambda x: x[0], res))
            values[col] = values[col].union(set(res))
    for k in values.keys():
        values[k] = list(values[k])
    return values


"""
Helper methods to determine various taxonomy rules.
"""


def is_attr_enhance(select_tokens1, select_tokens2, get_exact=False):
    '''
    Checks if attribute is "enhanced"
    '''
    contained_in_tokens2 = True
    for tok1 in select_tokens1:
        contained_in_tokens2 = contained_in_tokens2 and tok1 in select_tokens2
    is_enhance = contained_in_tokens2 and len(
        select_tokens1) < len(select_tokens2)
    if get_exact:
        if is_enhance:
            new_tokens = []
            for tok2 in select_tokens2:
                if tok2 not in select_tokens1:
                    new_tokens.append(tok2)
            return new_tokens
        return None
    return is_enhance


def is_attr_generalize(select_tokens1, select_tokens2, get_exact=False):
    '''
    Checks if attribute is "generalize"
    '''
    contained_in_tokens1 = True
    for tok2 in select_tokens2:
        contained_in_tokens1 = contained_in_tokens1 and tok2 in select_tokens1
    is_generalize = contained_in_tokens1 and len(
        select_tokens2) < len(select_tokens1)
    if get_exact:
        if is_generalize:
            removed_tokens = []
            for tok1 in select_tokens1:
                if tok1 not in select_tokens2:
                    removed_tokens.append(tok1)
            return removed_tokens
        return None
    return is_generalize


def is_attr_pivot(select_tokens1, select_tokens2, get_exact=False):
    '''
    Checks if attribute is "pivot"
    '''
    toks1 = set(select_tokens1)
    toks2 = set(select_tokens2)
    intersection = toks1.intersection(toks2)
    is_pivot = len(select_tokens1) == len(select_tokens2) and len(
        intersection) == len(select_tokens1) - 1
    if get_exact:
        if is_pivot:
            new_tokens = toks2.difference(intersection)
            new_tokens = list(new_tokens)
            return new_tokens
        return None
    return is_pivot


def is_same_select(select_tokens1, select_tokens2, get_exact=False):
    '''
    Checks if the attributes haven't changed.
    '''
    toks1 = set(select_tokens1)
    toks2 = set(select_tokens2)
    intersection = toks1.intersection(toks2)
    is_same = len(toks1) == len(toks2) and len(intersection) == len(toks1)
    if get_exact:
        if is_same:
            return []
        else:
            return None
    return is_same


def compare_where_clause(clause1, clause2):
    '''
    Returns true if clause1 and clause2 are the same. False otherwise.
    '''
    same = True
    for i in range(len(clause1)):
        same = same and clause1[i] == clause2[i]
    return same


def where_contains(where_clauses1, where_clauses2):
    '''
    Returns the clauses from where_clauses1 that are contained in where_clauses2
    '''
    contained_in_clauses2 = True
    for clause1 in where_clauses1:
        found_match = False
        for clause2 in where_clauses2:
            found_match = found_match or compare_where_clause(clause1, clause2)
        contained_in_clauses2 = contained_in_clauses2 and found_match
    return contained_in_clauses2


def is_predicate_add(clauses1, clauses2):
    '''
    Checks if predicate change is "add"
    '''
    return len(clauses1) < len(clauses2) and where_contains(clauses1, clauses2)


def is_predicate_generalize(clauses1, clauses2):
    '''
    Checks if predicate change is "generalize"
    '''
    return len(clauses1) > len(clauses2) and where_contains(clauses2, clauses1)


def is_predicate_same(clauses1, clauses2):
    '''
    Checks if predicates are the same.
    '''
    return len(clauses1) == len(clauses2) and \
        where_contains(clauses1, clauses2) and \
        where_contains(clauses2, clauses1)


def _get_predicate_difference_count(clauses1, clauses2):
    '''
    Count the number of unique predicates in each set. 
    Returns number of clauses not in clauses 2, number of clauses not in clauses1'''
    num_not_in_c2 = 0
    num_not_in_c1 = 0
    # Count num clauses contained from clauses1 to clauses2
    for c1 in clauses1:
        found_match = False
        for c2 in clauses2:
            found_match = found_match or compare_where_clause(c1, c2)
        if not found_match:
            num_not_in_c2 += 1
    # Reverse
    for c2 in clauses2:
        found_match = False
        for c1 in clauses1:
            found_match = found_match or compare_where_clause(c1, c2)
        if not found_match:
            num_not_in_c1 += 1
    return num_not_in_c2, num_not_in_c1


def is_predicate_pivot_singular(clauses1, clauses2):
    '''
    Checks if predicate change is "pivot" - pivot only one clause
    '''
    if len(clauses1) != len(clauses2):
        return False
    num_not_in_c2, num_not_in_c1 = _get_predicate_difference_count(
        clauses1, clauses2)
    return num_not_in_c1 == num_not_in_c2 and num_not_in_c2 == 1


def is_predicate_pivot_multiple(clauses1, clauses2):
    '''
    Checks if predicate change is "pivot" - pivot multiple clauses
    '''
    if len(clauses1) != len(clauses2):
        return False
    num_not_in_c2, num_not_in_c1 = _get_predicate_difference_count(
        clauses1, clauses2)
    return num_not_in_c1 == num_not_in_c2


def _select_arithmetic(select_tree):
    '''
    Removes aggregation and arithmetic terms
    '''
    # after getting an avg, min, count etc select,
    # it might have an 'add' 'sub' etc. clause
    if 'distinct' in select_tree.keys():
        return select_tree['distinct']
    elif 'sub' in select_tree.keys():
        return select_tree['sub']
    elif 'add' in select_tree.keys():
        return select_tree['add']
    return []


def _extract_single_select(select_tree):
    '''
    Extracts a single select statement and returns it. 
    Select_tree is the parse tree corresponding to the SELECT statement
    '''
    # Case 0: 'select' -> str
    if type(select_tree) is str:
        return [select_tree]
    # case 1: 'select' -> dict
    if type(select_tree) is dict:
        if 'value' in select_tree.keys():
            values = [select_tree['value']]
    # case 2: 'select' -> list
    if type(select_tree) is list:
        values = []
        for i in select_tree:
            if type(i) is dict:
                values.append(i['value'])
            elif type(i) is str:
                values.append(i)
            else:
                assert False, 'encountered non dict or str in list select'
    for i in range(len(values)):
        if type(values[i]) is dict:
            ops_to_check = ['count', 'avg', 'max',
                            'min', 'sum', 'add', 'sub', 'value']
            for op in ops_to_check:
                if type(values[i]) is dict and op in values[i].keys():
                    values[i] = values[i][op]
                    if type(values[i]) is dict:
                        values[i] = _select_arithmetic(values[i])
    # Now collapse all sublists
    final_values = []
    for v in values:
        if type(v) is str:
            final_values.append(v)
        elif type(v) is list:
            final_values = final_values + v
        elif isinstance(v, numbers.Number):
            final_values.append(str(v))
        else:
            assert False, values
    return final_values


def _extract_from_clause(parsed_tree):
    '''
    Extracts a single where clause from a sql parse tree. 
    '''
    if isinstance(parsed_tree, dict) and 'value' in parsed_tree.keys():
        return parsed_tree['value']
    if isinstance(parsed_tree, dict):
        return ''
    if isinstance(parsed_tree, list):
        return _extract_from_clause(parsed_tree[0])
    return parsed_tree


def extract_select_values(parsed_tree):
    '''
    returns all select attributes in the sql parse tree.
    '''
    if 'select' in parsed_tree.keys():
        return _extract_single_select(parsed_tree['select']), [_extract_from_clause(parsed_tree['from'])]
    elif 'select_distinct' in parsed_tree.keys():
        return _extract_single_select(parsed_tree['select_distinct']), [_extract_from_clause(parsed_tree['from'])]
    elif 'union' in parsed_tree.keys():
        res = [extract_select_values(parsed_tree['union'][i])
               for i in range(len(parsed_tree['union']))]
        values = [r[0] for r in res]
        froms = [r[1] for r in res]
        return list(itertools.chain.from_iterable(values)), list(itertools.chain.from_iterable(froms))
    elif 'intersect' in parsed_tree.keys():
        res = [extract_select_values(parsed_tree['intersect'][i])
               for i in range(len(parsed_tree['intersect']))]
        values = [r[0] for r in res]
        froms = [r[1] for r in res]
        return list(itertools.chain.from_iterable(values)), list(itertools.chain.from_iterable(froms))
    elif 'except' in parsed_tree.keys():
        res = [extract_select_values(parsed_tree['except'][i])
               for i in range(len(parsed_tree['except']))]
        values = [r[0] for r in res]
        froms = [r[1] for r in res]
        return list(itertools.chain.from_iterable(values)), list(itertools.chain.from_iterable(froms))
    elif 'from' in parsed_tree.keys():
        return extract_select_values(parsed_tree['from'])
    return [], []


def _extract_where_subclauses(where_clause, has_parent=False):
    '''
    extracts sql where structure and returns a list [attribute, op, value]
    '''
    if type(where_clause) is str:
        return where_clause
    # print()
    # pprint_tree(where_clause)
    assert len(list(where_clause.keys())
               ) == 1, 'Where clause with more than one operator'
    op = list(where_clause.keys())[0]
    if op == 'and' or op == 'or':
        subclauses = where_clause[op]
        clause = [_extract_where_subclauses(sub, True) for sub in subclauses]
    elif op == 'nin' or op == 'in' or op == 'between':
        # This is a very complicated case - just ignore it for now
        return None
    elif op in BINARY_OPS:
        operands = where_clause[op]
        assert len(operands) == 2, 'more than two operands'
        for i in range(2):
            if type(operands[i]) is dict:
                operand_curr = extract_select_values(operands[i])[0]
                # print(operands)
                if len(operand_curr) == 0:
                    # Case where it is a literal
                    if 'literal' in operands[i].keys():
                        operands[i] = [operands[i]['literal']]
                    else:
                        had_arithmetic = False
                        for a_op in ARITHMETIC_OPS:
                            had_arithmetic = had_arithmetic or \
                                a_op in operands[i].keys()
                            if a_op in operands[i].keys():
                                operands[i] = [
                                    f'{operands[i][a_op][0]}-{a_op}\
                                        -{operands[i][a_op][1]}']
                        assert had_arithmetic
                else:
                    operands[i] = operand_curr
                assert len(operands[i]) == 1, where_clause
                operands[i] = operands[i][0]
        clause = [op, operands[0], operands[1]]
        for c in clause:
            assert type(c) is str or isinstance(
                c, numbers.Number), f'clause {c} \
                    was not a string {where_clause}'
        if not has_parent:
            clause = [clause]
    else:
        assert False, f'Shouldnt ever get here... {op}, {where_clause}'
    return clause


def extract_where_clauses(parsed_tree):
    '''
    returns a list of extracted where / predicate clauses from the sql parse tree
    '''
    if 'where' not in parsed_tree.keys():
        return []
    where_tree = parsed_tree['where']
    clauses = _extract_where_subclauses(where_tree)
    if clauses is not None:
        return [c for c in clauses if c is not None]
    else:
        return []
