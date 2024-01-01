import json
from os.path import join
from torch.utils.data import Dataset
from src.config import cfg
import random

from collections import Counter

"""
Torch datasets for CoVis after performing data transformations.
"""
class CoVis(Dataset):
    '''
    Main CoVis dataset.
    '''

    # split keys
    train_split = 'TRAIN'
    dev_split = 'DEV'
    test_split = 'TEST'

    def __init__(self, split, data_root=cfg.COVIS_ROOT, context_sentences=-1,
                 utterance_transforms=None,
                 column_transforms=None,
                 where_transforms=None,
                 seperate_context=False,
                 shuffle_columns=False,
                 oversampling=True,
                 oversampling_target="select"):
        # Context sentences is the number of previous utterances to keep
        self._split = split
        self.context = context_sentences
        self.previous_utterances = None
        self.utterances = None
        self.select_targets = None
        self.where_targets = None
        self.column_names = None
        self.debug_info = None
        self.utterance_transforms = utterance_transforms
        self.column_transforms = column_transforms
        self.where_transforms = where_transforms
        self.seperate_context = seperate_context
        self.shuffle_columns = shuffle_columns
        self.oversampling = oversampling
        self.oversampling_target = oversampling_target
        self._load_data(split, data_root)

    def _shuffle_columns(self, columns, trg_col):
        #Data augmentation by shuffling columns. 
        idxs = [i for i in range(len(columns))]
        random.shuffle(idxs)
        shuffled_cols = [columns[i] for i in idxs]
        return shuffled_cols, idxs[trg_col]

    def _over_sampling(self, raw_data):

        # We only oversample training set.
        if self._split != self.train_split:
            return raw_data

        # We only oversample specific tasks
        if self.oversampling_target not in ['select-count', 'predicate-operator', 'predicate-count']:
            print("The target does not require oversampling.")
            return raw_data


        if self.oversampling_target == "select-count" or self.oversampling_target == "select-column":
            print("Oversampling according to the count of selected attributes.")
            print(f"Before oversampling, counts of all instances are {dict(Counter([str(len(set(i['select_target']))) for i in raw_data]))}")

            distribution_select_target_count = dict(Counter([str(len(set(i['select_target']))) for i in raw_data]))

            max_select_target_count_value = max(distribution_select_target_count.values())

            oversampled_raw_data = []
            
            # We oversample all instances directly considering their inversed probability
            for i in raw_data:

                freq = distribution_select_target_count[str(len(set(i['select_target'])))]

                # if the oversampling rate of an item is larger than 100, we only duplicate it for 100 times.
                oversample_rate = min([round(max_select_target_count_value/freq), 100])

                if oversample_rate > 1:
                    oversampled_raw_data.extend([i]*oversample_rate)
                else:
                    if random.random() < (max_select_target_count_value/freq):
                        oversampled_raw_data.extend([i])

            print(f"After oversampling, counts of all instances are {dict(Counter([str(len(set(i['select_target']))) for i in oversampled_raw_data]))}")

        elif self.oversampling_target == "predicate-operator":
            print("Oversampling according to predicates' operators.")

            clauses = [i['where_clauses'] for i in raw_data]
            individual_clauses = []

            for c in clauses:
                individual_clauses.extend(c)

            print(f"Before oversampling, counts of all instances are {dict(Counter([str(c[1]) for c in individual_clauses]))}")

            distribution_predicate_operator = dict(Counter([str(c[1]) for c in individual_clauses]))

            max_predicate_operator_count_value = max(distribution_predicate_operator.values())

            oversampled_raw_data = []
        
            for i in raw_data:

                freqs = [distribution_predicate_operator[str(c[1])] for c in i['where_clauses']]

                if len(freqs) == 0:
                    continue
                
                # oversample an utterance according to its max oversampling rate when it is smaller than 100
                # otherwise, duplicate it for 100 times
                max_freq = max_predicate_operator_count_value/min(freqs)
                
                oversample_rate = min([round(max_freq), 100])

                oversampled_raw_data.extend([i]*oversample_rate)

            oversampled_clauses = [i['where_clauses'] for i in oversampled_raw_data]
            oversampled_individual_clauses = []

            for c in oversampled_clauses:
                oversampled_individual_clauses.extend(c)

            print(f"After oversampling, counts of all instances are {dict(Counter([str(c[1]) for c in oversampled_individual_clauses]))}")

        elif self.oversampling_target == "predicate-count" or self.oversampling_target == "predicate-column":


            print("Oversampling according to the count of predicate attributes.")
            print(f"Before oversampling, counts of all instances are {dict(Counter([str(len(i['where_clauses'])) for i in raw_data]))}")


            distribution_predicate_target_count = dict(Counter([str(len(i['where_clauses'])) for i in raw_data]))

            max_predicate_target_count_value = max(distribution_predicate_target_count.values())

            oversampled_raw_data = []
            
            # We oversample all instances directly considering their inversed probability
            for i in raw_data:

                freq = distribution_predicate_target_count[str(len(i['where_clauses']))]

                # if the oversampling rate of an item is larger than 100, we only duplicate it for 100 times.
                oversample_rate = min([round(max_predicate_target_count_value/freq), 100])

                if oversample_rate > 1:
                    oversampled_raw_data.extend([i]*oversample_rate)
                else:
                    if random.random() < (max_predicate_target_count_value/freq):
                        oversampled_raw_data.extend([i])

            print(f"After oversampling, counts of all instances are {dict(Counter([str(len(i['where_clauses'])) for i in oversampled_raw_data]))}")
        
        return oversampled_raw_data

    def _load_data(self, split, data_root):
        fname = None
        if split == self.train_split:
            fname = cfg.TRAIN_FNAME
        elif split == self.test_split:
            fname = cfg.TEST_FNAME
        elif split == self.dev_split:
            fname = cfg.DEV_FNAME
        else:
            assert False, "You did not specify a proper CoVis split..."
        with open(join(data_root, fname), 'r') as fp:
            raw_data = json.load(fp)

        if self.oversampling:
            raw_data = self._over_sampling(raw_data)

        self.previous_utterances = [i['previous_utterances'] for i in raw_data]
        self.prior_select = [i['prior_select'] for i in raw_data]
        self.prior_predicate = [i['prior_where_col'] for i in raw_data]
        self.utterances = [i['utterance'] for i in raw_data]
        self.select_targets = [i['select_target'] for i in raw_data]
        self.column_names = [i['column_names'] for i in raw_data]
        self.where_targets = [i['where_clauses'] for i in raw_data]
        self.debug_info = [i['debug_info'] for i in raw_data]
        self.select_prior = [i['prior_select'] for i in raw_data]

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        columns = self.column_names[index]
        select_target = self.select_targets[index]
        where_target = self.where_targets[index]

        if self.shuffle_columns:
            columns, select_target = self._shuffle_columns(
                columns, select_target)

        context_num = self.context
        
        # if required context is smaller than existing context, only pick the latest ones
        if self.context != -1 and \
                self.context < len(self.previous_utterances[index]):
            context = self.previous_utterances[index][:self.context]
            context_num = self.context
        # if required context is larger than or equal to existing context, pick all context
        else:
            context_num = len(self.previous_utterances[index])
            context = self.previous_utterances[index]
        
        previous_context = {
            "attribute": [],
            "predicate": []
        }

        # check if context is actually empty
        try:
            if self.previous_utterances[index] != [""]:
                for cxt in range(0, context_num):
                    # All correct queries should have more than 1 selected attributes
                    if len(self.prior_select[index][cxt]) > 0:

                        previous_context['attribute'].extend(self.prior_select[index][cxt])

                        for predicate in self.prior_predicate[index][cxt]:
                            if predicate[0] == 'eq':
                                previous_context['predicate'].append(f"{predicate[1]} is {predicate[2]}")
                            elif predicate[0] == 'lt':
                                previous_context['predicate'].append(f"{predicate[1]} is less than {predicate[2]}")
                            elif predicate[0] == 'gt':
                                previous_context['predicate'].append(f"{predicate[1]} is larger than {predicate[2]}")
                            elif predicate[0] == 'neq':
                                previous_context['predicate'].append(f"{predicate[1]} is not {predicate[2]}")
        except:
            print(self.previous_utterances[index], self.prior_predicate[index])

        context_sentence = ""
        utterance = ""

        if len(previous_context['attribute']) > 0:
            context_sentence += "previous attributes: " +\
                ", ".join(list(dict.fromkeys(previous_context['attribute'])))
            context_sentence += ". [SEP] "
        
        if len(previous_context['predicate']) > 0:
            context_sentence += "previous predicates: " +\
                ", ".join(list(dict.fromkeys(previous_context['predicate'])))
            context_sentence += ". [SEP] "

        context_sentence += ' '
        
        if not self.seperate_context:
            utterance = context_sentence + self.utterances[index]
        else:
            utterance = self.utterances[index]

        previous_context = None
        
        if self.column_transforms is not None:
            for T in self.column_transforms:
                columns = T(columns)
        if self.utterance_transforms is not None:
            for T in self.utterance_transforms:
                utterance = T(utterance)
        if self.where_transforms is not None:
            for T in self.where_transforms:
                where_target = T(where_target)
        data_obj = {
            'utterance': utterance,
            'columns': columns,
            'select': list(set(select_target)),
            'where': where_target,
            'DEBUG': self.debug_info[index],
            'context': context,
            'previous_utterances': self.previous_utterances[index],
            'context_sentence': context_sentence
        }

        return data_obj