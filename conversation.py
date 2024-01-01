from contextlib import redirect_stdout, redirect_stderr

with open('eval_out.txt', 'w') as f:
            with redirect_stderr(f):
                with redirect_stdout(f):
                    import string
                    import pandas as pd
                    import warnings
                    from lux.vis.Vis import Vis, Clause

                    from torch.utils.data import Dataset
                    from src.datasets import transforms
                    from src.datasets.collate_functions import inference_collate
                    from torch.utils.data import DataLoader
                    import pytorch_lightning as pl
                    from transformers import logging as hf_logging
                    from tqdm import tqdm
                    from functools import partialmethod
                    from pipeline_inference import PipelineInference, KEYS
                    from time import time
                    
                    warnings.filterwarnings('ignore')
                    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
                    pl.seed_everything(5)
                    hf_logging.set_verbosity_error()

class CoVisInference(Dataset):
    '''
    CoVis dataset for demo usage.
    '''

    def __init__(self, context_sentences=-1,
                 utterance_transforms=None,
                 column_transforms=None,
                 where_transforms=None,
                 seperate_context=False):
        self.context = context_sentences
        self.previous_utterances = None
        self.previous_intent = None
        self.utterances = None
        self.select_targets = None
        self.where_targets = None
        self.column_names = None
        self.debug_info = None
        self.utterance_transforms = utterance_transforms
        self.column_transforms = column_transforms
        self.where_transforms = where_transforms
        self.seperate_context = seperate_context

    def load_one_utterance(self, previous_utterances, previous_intent, utterances, column_names):
        self.previous_utterances = [previous_utterances]
        self.previous_intent = previous_intent
        self.utterances = [utterances]
        self.column_names = [column_names]

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        columns = self.column_names[index]

        context_num = self.context
        
        # if required context is smaller than existing context, only pick the latest ones
        if self.context != -1 and \
                self.context < len(self.previous_intent["attribute"]):
            previous_context = self.previous_utterances[index][:self.context]
            context_num = self.context
        # if required context is larger than or equal to existing context, pick all context
        else:
            previous_context = self.previous_utterances[index]
            context_num = len(self.previous_intent["attribute"])
        
        previous_intent_prompt = {
            "attribute": [],
            "predicate": []
        }

        try:
            for cxt in range(0, context_num):
                # All correct queries should have more than 1 selected attributes
                if len(self.previous_intent["attribute"][cxt]) > 0:

                    previous_intent_prompt['attribute'].extend(self.previous_intent["attribute"][cxt])

                    for predicate in self.previous_intent["predicate"][cxt]:
                        if predicate[0] == '=':
                            previous_intent_prompt['predicate'].append(f"{predicate[1]} is {predicate[2]}")
                        elif predicate[0] == '<':
                            previous_intent_prompt['predicate'].append(f"{predicate[1]} is less than {predicate[2]}")
                        elif predicate[0] == '>':
                            previous_intent_prompt['predicate'].append(f"{predicate[1]} is larger than {predicate[2]}")
                        elif predicate[0] == '!=':
                            previous_intent_prompt['predicate'].append(f"{predicate[1]} is not {predicate[2]}")
        except:
            print(self.previous_intent)

        context_sentence = ""
        utterance = ""

            # lose predicate operator accuracy
        if len(previous_intent_prompt['attribute']) > 0:
            context_sentence += "previous attributes: " +\
                ", ".join(list(dict.fromkeys(previous_intent_prompt['attribute'])))
            context_sentence += ". [SEP] "
        
        if len(previous_intent_prompt['predicate']) > 0:
            context_sentence += "previous predicates: " +\
                ", ".join(list(dict.fromkeys(previous_intent_prompt['predicate'])))
            context_sentence += ". [SEP] "

        context_sentence += ' '
        
        if not self.seperate_context:
            utterance = context_sentence + self.utterances[index]
        else:
            utterance = self.utterances[index]


        print(utterance)

        if self.column_transforms is not None:
            for T in self.column_transforms:
                columns = T(columns)
        if self.utterance_transforms is not None:
            for T in self.utterance_transforms:
                utterance = T(utterance)
        data_obj = {
            'utterance': utterance,
            'columns': columns,
            'context': previous_context
        }

        return data_obj

#TODO: multi-thread optimization
class Conversation():
    
    def __init__(self, data_table, select_count=0, select_column=0, predicate_count=0,
                         predicate_column=0, predicate_operator=0, context = 1, debug_mode = False):
        
        self.data_table = data_table
        self.context = context
        self.debug_mode = debug_mode
        
        self.previous_utterances = []
        self.previous_intent = {
            "attribute": [],
            "predicate": []
        }

        self.current_utterance = ""
        self.current_intent = {
            "attribute": None,
            "predicate": None
        }
        
        self.model = PipelineInference(select_count=select_count, select_column=select_column, predicate_count=predicate_count,
                         predicate_column=predicate_column, predicate_operator=predicate_operator)
        
        self.columns = pd.Index(list(self.data_table.columns) + ['*'])
        
    def clean(self):
        self.previous_utterances = []
        self.previous_intent = {
            "attribute": [],
            "predicate": []
        }


        self.current_utterance = ""
        self.current_intent = {
            "attribute": None,
            "predicate": None
        }
    
    def clean_last(self):
        print(f"Remove {self.current_utterance}. Now previous_utterances include {self.previous_utterances}.")
        self.current_utterance = ""
        self.current_intent = {
            "attribute": None,
            "predicate": None
        }

    def ask(self, utterance):
        if self.current_utterance:
            self.previous_utterances.insert(0, self.current_utterance)
        
        # we consider that every valid intent should have attribute
        if self.current_intent['attribute']:
            self.previous_intent['attribute'].insert(0, self.current_intent['attribute'])
            self.previous_intent['predicate'].insert(0, self.current_intent['predicate'])
            
        self.current_utterance = utterance
        self.current_intent = {
            "attribute": None,
            "predicate": None
        }
        
        return self.run_pipeline()
        
    def mapping_idx_operator(self, operator_idx):
        if operator_idx == 0:
            return "="
        elif operator_idx == 1:
            return ">"
        elif operator_idx == 2:
            return "<"
        elif operator_idx == 3:
            return "!="
        else:
            print(operator_idx)

    
    def normalizedText(self, text):
        asciiAndNumbers = string.ascii_letters + string.digits
        # We are just allowing a-z, A-Z and 0-9 and use lowercase characters
        return ''.join(c for c in text if c in asciiAndNumbers).lower()


    def intent_to_lux_intent(self, intent):
        
        # We append the current intent to self.previous_intent at this step.

        title = []
        
        list_lux_intent = []
        
        if type(intent[KEYS.SELECT_COL]) == list:
            list_lux_intent += list(self.columns[intent[KEYS.SELECT_COL]])
            self.current_intent["attribute"] = list(self.columns[intent[KEYS.SELECT_COL]])
        else:
            list_lux_intent += [self.columns[intent[KEYS.SELECT_COL]]]
            self.current_intent["attribute"] = [self.columns[intent[KEYS.SELECT_COL]]]

        # if "*" notin list_lux_intent:

        star_in_intent = "*" in list_lux_intent
            
        list_predicate = []
        if intent[KEYS.PREDICATE_COUNT] > 0:
            for i in range(intent[KEYS.PREDICATE_COUNT]):
                if intent[KEYS.PREDICATE_VALUE][i]:
                    
                    predicate_column = self.columns[intent[KEYS.PREDICATE_COL][i]]
                    predicate_operator = self.mapping_idx_operator(intent[KEYS.PREDICATE_OP][i])
                    predicate_value = intent[KEYS.PREDICATE_VALUE][i]

                    # change the first predicate to attribute if "*" exists and "*" is the only column in selection, e.g., SELECT count(*) from
                    if (not star_in_intent) or (i > 0) or (intent[KEYS.SELECT_COUNT] > 1):
                        list_predicate.append([predicate_operator, predicate_column, predicate_value])

                        # recover from the lowered words in the predicate value inference process
                        if not(("float" in str(self.data_table[predicate_column].dtype)) or ("int" in str(self.data_table[predicate_column].dtype))):
                            for value in set(self.data_table[predicate_column]):
                                if self.normalizedText(value) == self.normalizedText(predicate_value):
                                    predicate_value = value


                        list_lux_intent.append(Clause(attribute=predicate_column, 
                                                    filter_op=predicate_operator, value=predicate_value))
                        
                        if "int" in str(self.data_table[predicate_column].dtype):
                            str_predicate_value = str(round(predicate_value))          
                        elif "float" in str(self.data_table[predicate_column].dtype):
                            str_predicate_value = str(predicate_value)       
                        else:
                            str_predicate_value = predicate_value

                        title.append(f"{predicate_column} {predicate_operator} {str_predicate_value}")
                    
                    else:
                        list_lux_intent.append(predicate_column)
                        self.current_intent["attribute"].append(predicate_column)
                
                # if the predicate value is not found in the data table, we simply use it as an selected attribute
                else:
                    predicate_column = self.columns[intent[KEYS.PREDICATE_COL][i]]
                    list_lux_intent.append(predicate_column)
                    self.current_intent["attribute"].append(predicate_column)
            
        self.current_intent['predicate'] = list_predicate

        list_lux_intent = [i for i in list_lux_intent if i != "*"]

        if self.debug_mode:
            print(f"Original intent: {intent}")
            print(f"Lux intent: {list_lux_intent}")
            print(f"Previous intent: {self.previous_intent}")
            print(f"Current intent: {self.current_intent}")
        
        return list_lux_intent, ", ".join(title)
        
    def run_pipeline(self):
        
        time_0 = time()

        model_dset = CoVisInference(
                               utterance_transforms=transforms.
                               default_utterance_transforms(),
                               where_transforms=transforms.default_where_transform(),
                               column_transforms=transforms.default_column_transforms(),
                               context_sentences=self.context
                               )

        model_dset.load_one_utterance(self.previous_utterances, self.previous_intent, self.current_utterance, self.columns)


        value_dset = CoVisInference(context_sentences=self.context,
                           seperate_context=True)

        value_dset.load_one_utterance(self.previous_utterances, self.previous_intent, self.current_utterance, self.columns)
        
        model_dloader = DataLoader(
            model_dset,
            batch_size=1,
            shuffle=False,
            collate_fn=inference_collate,
            num_workers=4)
        
        with open('eval_out.txt', 'w') as f:
            with redirect_stderr(f):
                with redirect_stdout(f):
                    time_1 = time()
                    print(f"load data: {time_1-time_0}")

                    self.model.run_select_count(model_dloader)
                    self.model.run_select_column(model_dloader)

                    time_2 = time()
                    print(f"select attribute: {time_2-time_1}")

                    # Run predicate count prediction
                    self.model.run_predicate_count(model_dloader)

                    time_3 = time()
                    print(f"count predicate: {time_3-time_2}")

                    # Run predicate column prediction
                    self.model.run_predicate_column(model_dloader)

                    time_4 = time()
                    print(f"predict predicate attribute: {time_4-time_3}")

                    # Run predicate operator prediction
                    self.model.run_predicate_operator(model_dloader)

                    time_5 = time()
                    print(f"predict predicate operator: {time_5-time_4}")

                    # Run value matching
                    self.model.run_predicate_value(value_dset, self.data_table)

                    time_6 = time()
                    print(f"predict predicate value: {time_6-time_5}")
        
                    print(f"total_time: {time_6-time_0}")

        intent = self.model.results.results[0]
        
        self.current_intent = intent

        lux_intent, title = self.intent_to_lux_intent(intent)
        
        vis = Vis(lux_intent, self.data_table, title=title)

        return vis