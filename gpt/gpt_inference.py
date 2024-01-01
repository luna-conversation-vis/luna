#!/usr/bin/env python
# coding: utf-8
from litellm import completion
import os
import json
from copy import deepcopy
from tqdm import tqdm
import sqlite3
import argparse
import re

def str_is_num(s):
    '''
    Checks if s is a number in string form.
    '''
    if s is None:
        return False
    try:
        _t = float(s)
        return True
    except ValueError:
        return False

def PUNCTUATION_STRIPPER(s):
    '''Strips all punctuation from string s.'''
    s = re.sub('\s\s+', ' ', re.sub(r'[^\w\s\.]', '', s))
    # remove periods for words
    matches = re.findall('[a-z]+\.', s)
    for m in matches:
        s = s.replace(m, m[:-1])
    return s

def compare_values(pred, gt):
    '''
    Compares the value portion of a single predicate.
    '''
    # returns true iff value1 == value2 after normalizing
    if pred is None:
        return False
    if isinstance(gt, str):
        gt = PUNCTUATION_STRIPPER(gt.lower().strip())
        if str_is_num(gt):
            gt = float(gt)
    
    if isinstance(pred, str):
        pred = PUNCTUATION_STRIPPER(pred.lower().strip())
        if str_is_num(pred):
            pred = float(pred)

    if pred == gt:
        return True
    else:
        return False


# In[24]:


## set ENV variables 
os.environ["AZURE_API_KEY"] = "PLEASE ENTER YOUR API KEY" 
os.environ["AZURE_API_BASE"] = "PLEASE ENTER YOUR API BASE"
os.environ["AZURE_API_TYPE"] = "azure" # [OPTIONAL] 
os.environ["AZURE_API_VERSION"] = "2023-07-01-preview" # [OPTIONAL]



# In[25]:


def initialize(cols):
    
    default_prompt_part1 = """
    You are a great assistant at identifying users' intent for visualizing a dataset from conversations.
    You should identify users' intent from two perspectives: (1) their interested data columns; (2) their interested data filters.
    Each data filter contains three parts: data column, data operator, and data value. 
    The valid operators in a data filter include: >, <, =, !=.
    Please infer the visualization intent from users' questions.
    When there are multiple questions in the context, they should be considered with the latest question.
    The dataset to be visualized include columns: 
    """
    default_prompt_part2 = """
    The visualization intent should be returned in JSON format:
    {
    "data columns": ["Column_A", "Column_B", ...],
    "data filters": [{
    "column": "Column_C",
    "operator": ">",
    "value": "Value_C"
    },
    {
    "column": "Column_D",
    "operator": "=",
    "value": "Value_D"
    }]
    }

    """
    
    
    messages = [{"content":default_prompt_part1 + cols + default_prompt_part2, "role": "system" }]
    # openai call
    response = completion(model="azure/luna", messages=messages)
    
    response_message = response['choices'][0]['message']
    
    messages.append(response_message)
    
    return response_message['content'], messages


def ask(question, messages):
    new_message = {
        "content": question,
        "role": "user"
    }
    
    messages.append(new_message)
    
    try:
    
        response = completion(model="azure/luna", messages=messages)

        response_message = response['choices'][0]['message']

        messages.append(response_message)
        
        return response_message['content'], messages
        
    except Exception as e:
        
        print(e)

        messages.append({"role": "assistant", "content": ""})

        
        return {
            "data columns": [],
            "data filters": []
        }, messages


def check_primary_key(key, db_id):
    
    #find all primary keys
    
    db_path = f"../data/cosql_dataset/database/{db_id}"

    sql_file = f"{db_path}/{db_id}.sqlite"

    conn = sqlite3.connect(sql_file)
    cursor = conn.cursor()
    # get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    res = cursor.fetchall()
    table_names = map(lambda x: x[0], res)

    pk = []

    for i in table_names:
        cursor.execute(f'SELECT l.name FROM pragma_table_info("{i}") as l WHERE l.pk <> 0;')

        res = cursor.fetchall()

        pk_names = map(lambda x: x[0], res)

        pk += (list(pk_names))

    
    
    # clean names
    
    pk = [i.replace("_", " ").lower() for i in pk]
    
    # check if the key is a primary key
    
    if key in pk:
        return True
    else:
        return False


def compute_acc(data):
    select_correct_count = 0
    filter_correct_count = 0
    all_correct_count = 0
    all_wrong_count = 0
    
    parse_fail = 0

    incomplete_filter = 0
    incomplete_json = 0
    
    incomplete_filter_flag = 0
    incomplete_json_flag = 0
    
    select_correct_filter_wrong = 0
    select_wrong_filter_correct = 0


    num_data = 0

    list_dict_result = []

    for d_ in data:
        d = deepcopy(d_)
        if 'result' not in d:
            continue
        else:

            d['result'].append(d['pred_intent'])

            # reverse the results to keep consistent with 'prior_select' and 'prior_where_col'
            results = list(reversed(d['result']))

            # add current utterance and ground_truth into the stack
            currect_select = list(set([d['column_names'][i] for i in d['select_target']]))

            d['prior_select'].insert(0, currect_select)

            current_filter = [[i[1], d['column_names'][i[0]], i[2]] for i in d['where_clauses']]

            d['prior_where_col'].insert(0, current_filter)

            d['previous_utterances'].insert(0, d['utterance'])

            if len(results) < len(d['prior_select']):
                continue

            for idx, select in enumerate(d['prior_select']):
                
                if d['previous_utterances'][idx] == "":
                    continue

                num_data += 1
                incomplete_json_flag = 0
                incomplete_filter_flag = 0

                dict_result = {
                    "select_correct": False,
                    "filter_correct": False,
                    "utterance": d['previous_utterances'][idx]
                }


                try:
                    if  "```json" in results[idx]:
                        results[idx] = results[idx].split("```json")[1].split("```")[0]
                    if "```" in results[idx]:
                        results[idx] = results[idx].split("```")[1].split("```")[0]
                        
                    if "{" in results[idx]:
                        temp = results[idx].split("{")[1:]
                        temp_str = "{".join(temp)
                        results[idx] = "{" + temp_str
                    
                    result = json.loads(results[idx])
                except Exception as e:
                    parse_fail += 1
                    continue

                if "data columns" not in result:
                    incomplete_json += 1
                    incomplete_json_flag = 1
                else:
                    
                    result_columns = result["data columns"]
                    
                    if set(select) == (set(result_columns)):
                        dict_result['select_correct'] = True
                        select_correct_count += 1
                    
                    elif "*" in select and len(select) == 1:
                        
                        result_columns = result["data columns"]
                        
                        db_id = d['debug_info']['db_id']
                        
                        if len(result_columns) == 1:
                            if check_primary_key(result_columns[0], db_id):
                                dict_result['select_correct'] = True
                                select_correct_count += 1
                    
                    else:
                        if "*" in select: select = [i for i in select if i != "*"]

                        result_columns = [i for i in result["data columns"] if i != "*"]


                        if set(select) == (set(result_columns)):
                            dict_result['select_correct'] = True
                            select_correct_count += 1
                
                if incomplete_json_flag:
                    continue
                
                if "data filters" not in result:
                    incomplete_json += 1

                else:
                    dict_op = {
                        "eq": "=",
                        "neq": "!=",
                        "gt": ">",
                        "lt": "<"
                    }

                    if len(d['prior_where_col'][idx]) == len(result['data filters']):
                        val_correct = True
                        for c1 in d['prior_where_col'][idx]:
                            match = False
                            for c2 in result['data filters']:

                                c2_dict = c2

                                if ("value" not in c2_dict) or ("operator" not in c2_dict) or ("column" not in c2_dict):
                                    incomplete_filter += 1
                                    incomplete_filter_flag = 1
                                    continue
                                else:
                                    if dict_op[c1[0]] == c2_dict['operator'] and c1[1] == c2_dict['column']\
                                        and compare_values(c2_dict['value'], c1[2]):
                                        match = True
                            val_correct = val_correct and match


                        if val_correct:
                            dict_result['filter_correct'] = True
                            filter_correct_count += 1
                    
                    if incomplete_filter_flag:
                        continue
                    else:

                        list_dict_result.append(dict_result)

                        if dict_result['filter_correct'] and dict_result['select_correct']:
                            all_correct_count += 1
                        elif dict_result['filter_correct'] and not dict_result['select_correct']:
                            select_wrong_filter_correct += 1
                        elif not dict_result['filter_correct'] and dict_result['select_correct']:
                            select_correct_filter_wrong += 1
                        else:
                            all_wrong_count += 1

    filter_correct_count = sum(map(lambda x: x['filter_correct'], list_dict_result))
    select_correct_count = sum(map(lambda x: x['select_correct'], list_dict_result))

    print("Overall accuracy: ", all_correct_count/num_data)
    print("Correct attr only: ", select_correct_filter_wrong/num_data)
    print("Correct filter only: ", select_wrong_filter_correct/num_data)
    print("Wrong attr + filter: ", all_wrong_count/num_data)
    print("Other errors: ", (parse_fail+incomplete_filter+incomplete_json)/num_data)

    return all_correct_count/num_data, select_correct_filter_wrong/num_data, select_wrong_filter_correct/num_data, all_wrong_count/num_data, (parse_fail+incomplete_filter+incomplete_json)/num_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['train', 'val', 'test'], type=str)
    parser.add_argument('--re_run', default=0, type=int)
    parser.add_argument('--model_name', choices=['GPT-4', 'GPT-3.5'], type=str)

    args = parser.parse_args()

    if args.re_run:

        if args.split == "val":
            data = json.load(open(f"../data/covis/dev_multi.json"))
        else:
            data = json.load(open(f"../data/covis/{args.split}_multi.json"))

        previous_ran_conversation = []

        results = []

        for d in tqdm(list(reversed(data))):
        #     previous_ran_conversation = d
            if d['utterance'] in previous_ran_conversation:
                continue
                
            cols = ", ".join(d['column_names'])
                
            res, messages = initialize(cols)

            if d['previous_utterances'][0] == "":
                
                new_data_obj = deepcopy(d)
                
                pred, messages = ask(d['utterance'], messages)
                d['result']= [pred]

                new_data_obj['previous_utterances'] = ['']
                new_data_obj['utterance'] = d['utterance']
                new_data_obj['pred_intent'] = pred

                results.append(new_data_obj)
            else:
                d['result'] = []
                asked_utterance = []
                for u in list(reversed(d['previous_utterances'])) + [d['utterance']]:
                    new_data_obj = deepcopy(d)
                    
                    pred, messages = ask(u, messages)
                    d['result'].append(pred)
                    
                    new_data_obj['previous_utterances'] = list(reversed(asked_utterance))
                    asked_utterance.append(u)
                    new_data_obj['utterance'] = u
                    new_data_obj['pred_intent'] = pred
                    
                    results.append(new_data_obj)
                    
            previous_ran_conversation = deepcopy(d['previous_utterances'])

        with open(f"./{args.model_name}/results_{args.split}.json", "w") as f:
            json.dump(results, f)


    compute_acc(json.load(open(f"./{args.model_name}/results_{args.split}.json", "r")))