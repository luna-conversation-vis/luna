from os.path import join
import json

"""
Helper script to split CoSQL splits [train, dev] into CoVis splits [train, dev, test]
"""

# Percent of the CoSQL dev to convert to the new dev and new test sets.
PERC_DEV = .65
PERC_TEST = .35

if __name__ == '__main__':
    print('CAUTION: YOU SHOULD NOT BE RUNNING THIS IN PYTHON. USE THE SHELL SCRIPT TO PREPARE DATA')
    cosql_directory = 'data/cosql_dataset/sql_state_tracking'

    # We keep the training set the same but change the dev set
    dev_file = '_cosql_dev.json'
    with open(join(cosql_directory, dev_file), 'r') as fp:
        standard_dev_obj = json.load(fp)

    num_interactions = len(standard_dev_obj)

    # Determine split
    num_dev = num_interactions * PERC_DEV
    num_test = num_interactions * PERC_TEST
    print(
        f'Attempting to split into {num_dev} for dev and {num_test} for test split')

    db_counts = {}
    db_dialgoue_mapping = {}
    for dialogue in standard_dev_obj:
        db_id = dialogue['database_id']
        if db_id not in db_counts.keys():
            db_counts[db_id] = 0
            db_dialgoue_mapping[db_id] = []
        db_counts[db_id] += 1
        db_dialgoue_mapping[db_id].append(dialogue)
    print(len(list(db_counts.keys())))
    print(db_counts)

    dev_db_ids = []
    curr_dev_count = 0
    test_db_ids = []

    for k in db_counts.keys():
        if num_dev - (curr_dev_count + db_counts[k]) > 0:
            dev_db_ids.append(k)
            curr_dev_count += db_counts[k]
        else:
            test_db_ids.append(k)

    print(
        f'Using {PERC_DEV*100}% for dev set and {PERC_TEST*100}% for the test set.')
    print(f'Dev size {sum([db_counts[i] for i in dev_db_ids])}')
    print(f'Test size {sum([db_counts[i] for i in test_db_ids])}')
    dev_set = []
    test_set = []
    for k in db_dialgoue_mapping.keys():
        if k in dev_db_ids:
            dev_set = dev_set + db_dialgoue_mapping[k]
        elif k in test_db_ids:
            test_set = test_set + db_dialgoue_mapping[k]
        else:
            assert False, 'Found an unprocessed key'
    print(f'Final counts: Dev {len(dev_set)} Test {len(test_set)}')

    # Write new dev and test set
    with open(join(cosql_directory, 'cosql_dev.json'), 'w+') as fp:
        json.dump(dev_set, fp)
    with open(join(cosql_directory, 'cosql_test.json'), 'w+') as fp:
        json.dump(test_set, fp)
    print('Finished!')
