import sqlite3
from src.datasets.database import PUNCTUATION_STRIPPER, Database, Dataframe
from src.config import cfg
import json


class TemporalValueMatching:
    '''Final value matching scheme. Breaks matching according temporally (i.e., favors matches from more recent utterances).'''

    def __init__(self, table_md_location=cfg.TABLE_METADATA_FP, grams=-1, mode="last"):
        self.table_md_location = table_md_location
        self.table_md = None
        self.databases = {}
        self._grams = grams
        self.mode = mode
        self._load()

    def _load(self):
        with open(cfg.TABLE_METADATA_FP, 'r') as fp:
            self.table_md = json.load(fp)
        for md in self.table_md:
            id = md['db_id']
            self.databases[id] = Database(
                id, parent_directory=cfg.DATA_TABLES, schema=md, grams=self._grams)

    def str_is_num(self, s):
        if s is None:
            return False
        try:
            _t = float(s)
            return True
        except ValueError:
            return False

    def __call__(self, db_id, utterance, context, target_column):
        return self.inference(db_id=db_id, utterance=utterance, context=context, target_column=target_column)

    def inference(self, db_id, utterance, context, target_column):
        db = self.databases[db_id]
        current_uttr = utterance
        pred_value = None

        all_utterance = context + [current_uttr] if self.mode=="last" else reversed(context + [current_uttr])

        for utterance in all_utterance:
            utterance = PUNCTUATION_STRIPPER(utterance.lower().strip())
            try:
                matches = list(
                    set(db.search_column(utterance, target_column)[0]))
            except sqlite3.OperationalError:
                print(f'Ran into SQL Error - column', target_column, db_id)
                return None
            if len(matches) == 1:
                match = matches[0]
                pred_value = match
        if self.str_is_num(pred_value):
            pred_value = float(pred_value)
        return pred_value
    

class TemporalValueMatchingDataframe:
    '''Final value matching scheme. Breaks matching according temporally (i.e., favors matches from more recent utterances).'''

    def __init__(self, grams=-1):
        self._grams = grams

    def str_is_num(self, s):
        if s is None:
            return False
        try:
            _t = float(s)
            return True
        except ValueError:
            return False

    def __call__(self, df, utterance, context, target_column):
        return self.inference(df, utterance, context, target_column)

    def inference(self, df, utterance, context, target_column):

        dataframe = Dataframe(df, grams=self._grams)
        
        current_uttr = utterance
        pred_value = None
        for utterance in context + [current_uttr]:
            utterance = PUNCTUATION_STRIPPER(utterance.lower().strip())
            matches = list(
                    set(dataframe.search_column(utterance, target_column)[0]))
            if len(matches) == 1:
                match = matches[0]
                pred_value = match
        if self.str_is_num(pred_value):
            pred_value = float(pred_value)
        return pred_value