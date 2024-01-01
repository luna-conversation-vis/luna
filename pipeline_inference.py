import pickle
import torch
from src.models.builder import load_checkpoint
from src.datasets.database import PUNCTUATION_STRIPPER
from src.models.temporal_value_matcher import TemporalValueMatchingDataframe
from src.common.utils import batch_to_device, str_is_num
from typing import List
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
pl.seed_everything(5)

"""
The entire assembled pipeline. See argparse args to run.
"""


class KEYS:
    '''
    Keys for ResultsCache
    '''
    SELECT_COUNT = 'select count'
    SELECT_COL = 'select column'
    PREDICATE_COUNT = 'predicate count'
    PREDICATE_COL = 'predicate column'
    PREDICATE_OP = 'predicate operator'
    PREDICATE_VALUE = 'predicate value'


class ResultsCache:
    '''
    Stores results based on index in the dataset.
    Helpful for storing predictions insteading of having to keep all models in memory.
    '''

    def __init__(self, tmp_file=False):
        self.tmp_file = tmp_file
        self.index = 0
        self.key = None
        self._cache = []

    def set_key(self, key):
        self.key = key
        self.index = 0

    def write(self, results: List):
        for result in results:
            if self.index >= len(self._cache):
                self._cache.append({self.key: result})
            else:
                self._cache[self.index][self.key] = result
            self.index += 1

    def write_singular(self, result):
        self.write([result])

    @property
    def results(self):
        return self._cache

    def __getitem__(self, idx):
        return self._cache[idx]

    def __len__(self):
        return len(self._cache)

    def build_from_cache(self, old_cache):
        self._cache = old_cache

    def dump(self, fp):
        with open(fp, 'wb+') as _fp:
            pickle.dump(self._cache, _fp)


class PipelineInference:
    '''
    Full pipeline class. Stores the models to predict each task.
    '''
    # arguments are the v_num from pytorch_lightning

    def __init__(self, select_column=0,  # 15
                 select_count=0,
                 predicate_count=3,
                 predicate_column=19,
                 predicate_operator=2,
                 value_matcher=0,
                 device=torch.device("cuda" if torch.cuda.is_available()
                                     else "cpu"),
                 staggered=True):
        # NOTE Setting any of these to -1 will use the ground truth value
        self.select_count_id = select_count
        self.select_column_id = select_column
        self.predicate_count_id = predicate_count
        self.predicate_column_id = predicate_column
        self.predicate_operator_id = predicate_operator
        self.value_matcher_id = value_matcher
        self.select_count = None
        self.select_column = None
        self.predicate_count = None
        self.predicate_column = None
        self.predicate_operator = None
        self.value_matcher = None
        self._loaded_model_version = None
        self.device = device
        self.staggered = staggered
        self.results = ResultsCache()
        # Load once when initialization
        self._load_select_count()
        self._load_select_column()
        self._load_predicate_count()
        self._load_predicate_column()
        self._load_predicate_operator()
        self._load_value_matcher()

    def _load_select_count(self):
        # Load predicate-count
        if self.select_count_id != -1:
            self._loaded_model_version = self.select_count
            model_name = 'select_count'
            ckpt_fname = f'tb_logs/select-count/version_{self.select_count_id}/checkpoints/best_model.ckpt'
            self.select_count = load_checkpoint(model_name, ckpt_fname)
            self.select_count = self.select_count.to(self.device).eval()

    def _load_select_column(self):
        # Load predicate-count
        if self.select_column_id != -1:
            self._loaded_model_version = self.select_column
            model_name = 'select_col'
            ckpt_fname = f'tb_logs/select-column/version_{self.select_column_id}/checkpoints/best_model.ckpt'
            self.select_column = load_checkpoint(model_name, ckpt_fname)
            self.select_column = self.select_column.to(self.device).eval()

    def _load_predicate_count(self):
        # Load predicate-count
        if self.predicate_count_id != -1:
            self._loaded_model_version = self.predicate_count_id
            model_name = 'where_count'
            ckpt_fname = f'tb_logs/predicate-count/version_{self.predicate_count_id}/checkpoints/best_model.ckpt'
            self.predicate_count = load_checkpoint(model_name, ckpt_fname)
            self.predicate_count = self.predicate_count.to(self.device).eval()

    def _load_predicate_column(self):
        # load predicate-column
        if self.predicate_column_id != -1:
            self._loaded_model_version = self.predicate_column_id
            model_name = 'where_col'
            ckpt_fname = f'tb_logs/predicate-column/version_{self.predicate_column_id}/checkpoints/best_model.ckpt'
            self.predicate_column = load_checkpoint(
                model_name, ckpt_fname).to(self.device).eval()

    def _load_predicate_operator(self):
        # load predicate-operator
        if self.predicate_operator_id != -1:
            self._loaded_model_version = self.predicate_operator_id
            model_name = 'where_operator'
            ckpt_fname = f'tb_logs/predicate-operator/version_{self.predicate_operator_id}/checkpoints/best_model.ckpt'
            self.predicate_operator = load_checkpoint(
                model_name, ckpt_fname).to(self.device).eval()

    def _load_value_matcher(self):
        # load value matcher
        if self.value_matcher_id != -1:
            self._loaded_model_version = self.value_matcher
            self.value_matcher = TemporalValueMatchingDataframe()

    def run_select_count(self, loader):
        print('Loading select count prediction network...')
        self.results.set_key(KEYS.SELECT_COUNT)
        print('Done loading. Running inference...')
        for batch in tqdm(loader):
            if self.select_count == -1:
                trgs = batch[2]['select']
                for trg in trgs:
                    self.results.write_singular(len(trg))
            else:
                with torch.no_grad():
                    batch = batch_to_device(batch, self.device)
                    # N pts x 5
                    model_out = self.select_count(batch)
                    model_out = torch.argmax(model_out, dim=1).cpu().tolist()
                    self.results.write(model_out)
        print('Done select count prediction!')

    def run_select_column(self, loader):
        print('Loading select column prediction network...')
        self.results.set_key(KEYS.SELECT_COL)

        print('Done loading. Running inference...')
        for batch in tqdm(loader):
            if self.select_column == -1:
                trgs = batch[2]['select']
                for trg in trgs:
                    self.results.write_singular([t[0] for t in trg])
            else:
                with torch.no_grad():
                    batch = batch_to_device(batch, self.device)
                    # N pts x L max columns
                    model_out = self.select_column(batch)

                    model_out = torch.topk(
                        model_out, largest=True, sorted=True, k=6).indices.cpu().tolist()

                    self.results.write(model_out)

        # use the count value to do selection
        for x in self.results.results:
            count = x[KEYS.SELECT_COUNT]
            if count > 0:
                x[KEYS.SELECT_COL] = x[KEYS.SELECT_COL][:count]
            else:
                x[KEYS.SELECT_COL] = []
        
        print('Done predicting select columns!')

    def run_predicate_count(self, loader):
        print('Loading predicate count prediction network...')
        
        self.results.set_key(KEYS.PREDICATE_COUNT)
        print('Done loading. Running inference...')
        for batch in tqdm(loader):
            if self.predicate_count == -1:
                trgs = batch[2]['where']
                for trg in trgs:
                    self.results.write_singular(len(trg))
            else:
                with torch.no_grad():
                    batch = batch_to_device(batch, self.device)
                    # N pts x 5
                    model_out = self.predicate_count(batch)
                    model_out = torch.argmax(model_out, dim=1).cpu().tolist()
                    self.results.write(model_out)
        print('Done predicate count prediction!')

    def run_predicate_column(self, loader):
        print('Loading predicate column prediction network...')
        
        self.results.set_key(KEYS.PREDICATE_COL)

        print('Done loading. Running inference...')
        for batch in tqdm(loader):
            if self.predicate_column == -1:
                trgs = batch[2]['where']
                for trg in trgs:
                    self.results.write_singular([t[0] for t in trg])
            else:
                with torch.no_grad():
                    batch = batch_to_device(batch, self.device)
                    # N pts x L max columns
                    model_out = self.predicate_column(batch)
                    model_out = torch.topk(
                        model_out, largest=True, sorted=True, k=5).indices.cpu().tolist()

                    self.results.write(model_out)
        
        print('Done predicting predicate columns!')

    def run_predicate_operator(self, loader: DataLoader):
        print('Loading predicate operator prediction network...')
        
        self.results.set_key(KEYS.PREDICATE_OP)
        print('Done loading. Finalizing predicate column predictions for input')
        for x in self.results.results:
            count = x[KEYS.PREDICATE_COUNT]
            if count > 0:
                x[KEYS.PREDICATE_COL] = x[KEYS.PREDICATE_COL][:count]
            else:
                x[KEYS.PREDICATE_COL] = []
        print('Done computing relevant columns.')
        print(f'Using a batch size of {loader.batch_size}.')
        print('Done loading. Running inference...')
        results_index = 0
        for batch in tqdm(loader):
            if self.predicate_operator == -1:
                trgs = batch[2]['where']
                for trg in trgs:
                    self.results.write_singular([t[1] for t in trg])
            else:
                # First grab predicted columns and counts
                predicted_counts = []
                predicted_columns = []
                for _ in range(loader.batch_size):
                    if results_index < len(self.results):
                        predicted_counts.append(
                            self.results[results_index][KEYS.PREDICATE_COUNT])
                        predicted_columns.append(
                            self.results[results_index][KEYS.PREDICATE_COL])
                        results_index += 1
                with torch.no_grad():
                    batch = batch_to_device(batch, self.device)
                    # Num predicates
                    model_out = self.predicate_operator(
                        batch, use_gt=False, column_predictions=predicted_columns)
                    model_out = torch.argmax(model_out, dim=1).cpu()
                    index = 0
                    for c in predicted_counts:
                        data_pt_predictions = []
                        for _ in range(c):
                            data_pt_predictions.append(model_out[index].tolist())
                            index += 1
                        self.results.write_singular(data_pt_predictions)
        print('Done predicting operators!')

    def run_predicate_value(self, dataset, df):
        print('Loading value matcher and relevant databases...')
        
        self.results.set_key(KEYS.PREDICATE_VALUE)
        print('Done loading. Performing matching...')
        for pt, result_so_far in tqdm(zip(dataset, self.results.results), total=len(dataset)):
            if self.value_matcher == -1:
                gt = [t[-1] for t in pt['where']]
                for i in range(len(gt)):
                    if isinstance(gt[i], str):
                        gt[i] = PUNCTUATION_STRIPPER(gt[i].lower().strip())
                        if str_is_num(gt[i]):
                            gt[i] = float(gt[i])
                self.results.write_singular(gt)
            else:
                utterance = pt['utterance']
                context = pt['context']
                columns = pt['columns']
                # db_id = pt['DEBUG']['db_id']
                predicted_values = []
                for col_index in result_so_far[KEYS.PREDICATE_COL]:
                    predicted_values.append(self.value_matcher(
                            df, utterance, context, columns[col_index]))
  
                self.results.write_singular(predicted_values)
        print('Done matching values!')


def compare_values(pred, gt):
    '''
    Compares the value portion of a single predicate.
    '''
    if pred is None:
        return False
    if isinstance(gt, str):
        gt = PUNCTUATION_STRIPPER(gt.lower().strip())
        if str_is_num(gt):
            gt = float(gt)
    if pred == gt:
        return True
    else:
        return False
