from mo_sql_parsing import parse
from tqdm import tqdm
import json
import argparse
import re
from src.datasets import cosql
from src.datasets.cosql import build_corpus, \
    flatten_corpus, \
    build_database_metadata, \
    extract_select_values, \
    extract_where_clauses
from src.config import cfg, WhereCategories
import pytorch_lightning as pl
from typing import Dict, List

pl.seed_everything(5)

"""
Script for mapping CoSQL to CoVis. Run this via "generate-covis-splits.sh".
Requires that CoSQL [train, dev] already split into [train, dev, test]. Use "prepare-datasets.sh" for this
"""


def build_data_obj(prev_utterances, utterance,
                   select_target, where_clauses, cols, prior_select, prior_where_col,
                   **kwargs):
    '''
    Final CoVis data object builder
    prior_where_col now updates to full predicates instead of cols only
    '''
    return {
        'previous_utterances': prev_utterances,
        'utterance': utterance,
        'select_target': select_target,
        'column_names': cols,
        'where_clauses': where_clauses,
        'debug_info': kwargs,
        'prior_select': prior_select,
        'prior_where_col': prior_where_col,
    }


def return_unique_cols(original, english):
    '''
    Given a set of columns (e.g., 'student_ids'), returns unique columns 
    and the english mapping (e.g., 'student_ids' -> 'student ids')
    '''
    # Returns a list of the english columns and a function
    # to map an index to the new space
    unique_english = {}
    for i, val in enumerate(english):
        if val not in unique_english.keys():
            unique_english[val] = []
        unique_english[val].append(i)
    new_mapping = {}
    for i, val in enumerate(unique_english.keys()):
        new_mapping[val] = i
    collapsed_columns = [None for _ in range(len(new_mapping.keys()))]
    for val in unique_english.keys():
        collapsed_columns[new_mapping[val]] = val
    for col in collapsed_columns:
        assert col is not None, 'Column mapped to None'

    def map_to_new_space(x): return new_mapping[english[original.index(x)]]
    return collapsed_columns, map_to_new_space


def query_fixes(query: str):
    '''
    Misc fixes to ensure MoSql parser can parse.
    '''
    query = query.replace('! =', '!=')
    query = query.replace('> =', '>=')
    query = query.replace('< =', '<=')
    query = query.replace('WHEREClaim_Status_Name', 'WHERE Claim_Status_Name')
    query = re.sub(r'(LIMIT|limit)\s(\d\W)+\d', '', query)
    if query == 'SELECT countryname FROM countries WHERE countryid = 1 or countryid = 2 or countryid = 3 ) ':
        return query_fixes(
            'SELECT countryname FROM countries WHERE countryid = 1 or countryid = 2 or countryid = 3')
    return query


def get_where_categorization(prev, curr):
    '''
    Finds predicate shift categorization and returns it.
    '''
    if cosql.is_predicate_same(prev, curr):
        return WhereCategories.NO_CHANGE
    if cosql.is_predicate_pivot_singular(prev, curr):
        return WhereCategories.PIVOT_SINGULAR
    if cosql.is_predicate_pivot_multiple(prev, curr):
        return WhereCategories.PIVOT_MULTIPLE
    if cosql.is_predicate_add(prev, curr):
        return WhereCategories.ADD
    if cosql.is_predicate_generalize(prev, curr):
        return WhereCategories.GENERALIZE
    return WhereCategories.FAILED


def get_attr_categorization(prev, curr):
    '''
    Finds attribute shift categorization and returns it.
    '''
    if cosql.is_attr_enhance(prev, curr):
        return 'enhance'
    if cosql.is_attr_generalize(prev, curr):
        return 'generalize'
    if cosql.is_attr_pivot(prev, curr):
        return 'pivot'
    if cosql.is_same_select(prev, curr):
        return 'same'
    return 'failed'


def clean_where_clauses(where_clauses):
    '''
    Sanitizes where / predicate clauses
    '''
    # remove any aliasing going on
    for i in range(len(where_clauses)):
        
        where_clauses[i][1] = where_clauses[i][1].lower(
            ).strip().split('.')[-1]
    like_clauses = []
    num_where_removed = 0  # clauses removed
    num_where_modified = 0  # clauses modified (e.g., lte -> lt)
    for j in range(len(where_clauses)):
        if where_clauses[j][0] == 'like':
            num_where_removed += 1
            like_clauses.append(where_clauses[j])
        if where_clauses[j][0] == 'gte':
            num_where_modified += 1
            where_clauses[j][0] = 'gt'
        if where_clauses[j][0] == 'lte':
            num_where_modified += 1
            where_clauses[j][0] = 'lt'
        if where_clauses[j][0] == 'not_like':
            num_where_removed += 1
            like_clauses.append(where_clauses[j])
    for j in like_clauses:
        where_clauses.remove(j)
    return where_clauses, num_where_removed, num_where_modified


def _is_alias(obj: Dict):
    if "value" in obj.keys() and "name" in obj.keys():
        return obj['name'].lower().strip(), obj['value'].lower().strip()
    return None


def find_sql_aliases(parse_tree, aliases=dict()):
    '''
    Recusive method to find sql aliasing (SELECT ids AS x FROM students)
    '''
    # Case 1 parse_tree is a dict
    # First check if this is already a valid alias
    if isinstance(parse_tree, Dict):
        alias = _is_alias(parse_tree)
        if alias is not None:
            aliases[alias[0]] = alias[1]
            return aliases
        for k in parse_tree.keys():
            aliases = find_sql_aliases(parse_tree[k], aliases=aliases)
        return aliases
    # Case 2 parse_tree is a list
    if isinstance(parse_tree, List):
        for e in parse_tree:
            aliases = find_sql_aliases(e, aliases=aliases)
        return aliases
    return aliases


def convert_col_to_canonical(target, target_from, aliases, db_metadata, db_id):
    '''
    Returns canonicalized columns. 
    '''
    target = target.lower().strip().split('.')
    target_from = target_from.lower().strip()
    original_table_names = [db_metadata[db_id]['table_names'][i][0] for i in range(
        len(db_metadata[db_id]['table_names']))]

    if len(target) == 1:
        target = target[0]

        # First try to match by column name - use from if there are multiple matches
        if target == '*':
            return [-1, 0]
        cols_squashed = [db_metadata[db_id]['columns'][i]
                         for i in range(len(db_metadata[db_id]['columns']))]
        col_names_squashed = [col[1] for col in cols_squashed]
        matches = []
        for i, c in enumerate(col_names_squashed):
            if c == target:
                matches.append(i)
        if len(matches) == 1:
            matched_column = cols_squashed[matches[0]]
            table_idx = matched_column[0]
            table_columns = [db_metadata[db_id]['columns'][i][1] for i in range(len(
                db_metadata[db_id]['columns'])) if db_metadata[db_id]['columns'][i][0] == table_idx]
            target_indexed = [table_idx, table_columns.index(target)]
        else:
            # We will try to use the target_from but sometimes it fails due to joins
            try:
                assert target == '*' or target_from in original_table_names, target_from
                table_idx = original_table_names.index(target_from)
                cols = [db_metadata[db_id]['columns'][i][1] for i in range(len(
                    db_metadata[db_id]['columns'])) if db_metadata[db_id]['columns'][i][0] == table_idx]
                target_indexed = [table_idx, cols.index(target)]
            except ValueError:
                matches = [matches[0]]
                matched_column = cols_squashed[matches[0]]
                table_idx = matched_column[0]
                table_columns = [db_metadata[db_id]['columns'][i][1] for i in range(len(
                    db_metadata[db_id]['columns'])) if db_metadata[db_id]['columns'][i][0] == table_idx]
                target_indexed = [table_idx, table_columns.index(target)]

    else:
        assert target[0] in original_table_names or aliases[target[0]
                                                            ] in original_table_names, target
        # table index
        if target[0] in original_table_names:
            target_indexed = [
                original_table_names.index(target[0]), None]
        else:
            target_indexed = [original_table_names.index(
                aliases[target[0]]), None]

        # column index
        table_cols = [db_metadata[db_id]['columns'][i][1] for i in range(len(
            db_metadata[db_id]['columns'])) if db_metadata[db_id]['columns'][i][0] == target_indexed[0]]
        assert target[1] in table_cols, target
        target_indexed[1] = table_cols.index(
            target[1])
    return target_indexed

def convert_col_to_english(col_name, cols, english):
    cols = [i.lower() for i in cols]
    col_name_idx = cols.index(col_name)
    # print(cols, col_name, english[col_name_idx])
    return english[col_name_idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=cfg.COVIS_ROOT)
    parser.add_argument('--v', action='store_true', help='verbose')
    parser.add_argument('--no_categorization', action='store_true')
    parser.add_argument(
        '--split', choices=['train', 'test', 'dev'], default='train')
    args = parser.parse_args()

    corpus = build_corpus(args.split)
    flat_corpus = flatten_corpus(corpus)
    print(f'Building split: {args.split}')
    print('Number of dialogues:', len(corpus))
    print('Number of interactions', len(flat_corpus))

    db_metadata = build_database_metadata()
    where_categorization = not args.no_categorization

    malformed_queries = ['SELECT count ( interval )  FROM train',
                         'SELECT avg ( T2.rating )  FROM useracct WHERE T2.u_id' +
                         ' = 1 AS T1 JOIN review AS T2 ON T1.u_id   =   T2.u_id' +
                         ' GROUP BY T2.u_id',
                         'SELECT order_id,  from Order_Items group by order_id ' +
                         'order by sum ( order_quantity )  desc limit 1',
                         'SELECT dept_store_chain_id FROM department_stores ' +
                         'GROUP BY dept_store_chain_id ORDER BY count ( * )  ' +
                         'DESC LIMIT 2 except SELECT dept_store_chain_id FROM' +
                         ' department_stores GROUP BY dept_store_chain_id ORDER' +
                         ' BY count ( * )  DESC LIMIT 1',
                         'SELECT End FROM appointment ORDER BY START DESC LIMIT 1',
                         'SELECT t3.name FROM visitor AS t1 JOIN visit AS t2 ON t1.id   =   t2.visitor_id JOIN museum AS t3 ON t3.Museum_ID   =   t2.Museum_ID WHERE t3.open_year  <  2009 AND t1.name  =   ( SELECT t1.name FROM visitor AS t1 JOIN visit AS t2 ON t1.id   =   t2.visitor_id JOIN museum AS t3 ON t3.Museum_ID   =   t2.Museum_ID WHERE t3.open_year  <  2009 INTERSECT SELECT t1.name FROM visitor AS t1 JOIN visit AS t2 ON t1.id   =   t2.visitor_id JOIN museum AS t3 ON t3.Museum_ID   =   t2.Museum_ID WHERE t3.open_year  >  2011 ) '
                         ]

    attr_diffs = {
        'enhance': 0,
        'pivot': 0,
        'same': 0,
        'failed': 0,
        'generalize': 0
    }
    data_pts = []
    bad_vals = 0
    num_where_removed = 0
    num_where_modified = 0
    num_where_total = 0
    for d_count, d in enumerate(tqdm(corpus)):
        if d.db_id == 'baseball_1':
            continue

        list_cols = db_metadata[d.db_id]['original'] + ['*']
        list_cols_english = db_metadata[d.db_id]['transformed'] + ['*']

        # print(list_cols, list_cols_english)

        for i in range(len(d)):
            utterance, query = d[i]
            query = query_fixes(query)
            if query in malformed_queries:
                bad_vals += 1
                continue
            parse_tree = parse(query)
            select_tokens, select_froms = extract_select_values(parse_tree)
            # op, col, val
            where_clauses = extract_where_clauses(parse_tree)
            like_clauses = []
            num_where_total += len(where_clauses)
            where_clauses, curr_num_removed, curr_num_modified = \
                clean_where_clauses(where_clauses)
            num_where_removed += curr_num_removed
            num_where_modified += curr_num_modified

            # now get the previous clauses so we can categorize the change
            prior_select = []
            prior_select_all = []
            prev_where_clause_all = []
            if where_categorization:
                if 0 < i:
                    for j in range(i-1, -1, -1):
                        prev_query = d[j][1]
                        # print(d[i-1])
                        prev_where_clause = None
                        if prev_query not in malformed_queries:
                            prev_query = query_fixes(prev_query)
                            prev_query_parsed = parse(prev_query)
                            prev_select, _ = extract_select_values(prev_query_parsed)

                            if isinstance(prev_select, list):
                                prior_select = []
                                for t in prev_select:
                                    prior_select.append(convert_col_to_english(t.lower(
                                ).strip().split('.')[-1], list_cols, list_cols_english))
                            prior_select = list(dict.fromkeys(prior_select))

                            prev_where_clause = extract_where_clauses(
                                parse(prev_query))
                            prev_where_clause, _, _ = clean_where_clauses(
                                prev_where_clause)

                            # change the format of column names from "first_name" to "first name"
                            prev_where_clause = [[clause[0], convert_col_to_english(clause[1], list_cols, list_cols_english), clause[2]] for clause in prev_where_clause]

                            prior_select_all.append(prior_select)
                            prev_where_clause_all.append(prev_where_clause)
                        else:
                            prev_where_clause = []
                            prior_select_all.append(prior_select)
                            prev_where_clause_all.append(prev_where_clause)
                else:
                    prev_where_clause = []
                    prior_select_all.append(prior_select)
                    prev_where_clause_all.append(prev_where_clause)

            else:
                where_change_categorization = -1
            # print(prior_select_all)
            # check to make sure the target is a valid select
            select_target = select_tokens
            select_from = select_froms
            
            select_target = [t.lower().strip().split('.')[-1] for t in select_target]
            cols_original = db_metadata[d.db_id]['original'] + ['*']
            
            for t in select_target:
                assert t in cols_original, \
                    f'Found a target not in the cols {select_target}, {cols_original}'
            
            cols = db_metadata[d.db_id]['transformed'] + ['*']

            cols, col_mapper = return_unique_cols(cols_original, cols)
            index_of_select_target = [col_mapper(t) for t in select_target]

            if i > 0:
                prev_utterances = []
                j = 0
                while j < i:
                    j += 1
                    prev_utterances.append(d[i-j][0])
            else:
                prev_utterances = ['']
            where_clauses_cleaned = []
            for clause in where_clauses:
                clause = [clause[1].lower().strip().split('.')[-1],
                            clause[0], clause[2]]
                clause[0] = col_mapper(clause[0])
                where_clauses_cleaned.append(clause)
            # Make the data object
            data_pts.append(build_data_obj(prev_utterances=prev_utterances,
                                            utterance=utterance,
                                            select_target=index_of_select_target,
                                            where_clauses=where_clauses_cleaned,
                                            cols=cols,
                                            where_categorization=-1,
                                            db_id=d.db_id,
                                            raw_sql=query,
                                            number_predicates=len(
                                                where_clauses_cleaned),
                            prior_select=prior_select_all, prior_where_col=prev_where_clause_all))
    if args.v:
        print('Number of where clauses modified:', num_where_modified)
        print('Number of where clauses removed:', num_where_removed)
        print('Total number of where clauses:', num_where_total)

    with open(f'data/covis/{args.split}_multi.json', 'w+') as fp:
        json.dump(data_pts, fp, indent=4)

    print('Done!')

