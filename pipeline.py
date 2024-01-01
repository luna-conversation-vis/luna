import pickle
import argparse
import torch
from src.datasets.covis import CoVis
from src.datasets import transforms
from src.datasets.collate_functions import collate
from src.models.builder import load_checkpoint
from src.datasets.transforms import PredicateOperationEncodeTransform
from src.datasets.database import PUNCTUATION_STRIPPER
from src.models.temporal_value_matcher import TemporalValueMatching
from src.config import cfg
from src.common.utils import batch_to_device, str_is_num
from typing import List
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
pl.seed_everything(5)

from contextlib import redirect_stdout, redirect_stderr
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
"""
The entire assembled pipeline. See argparse args to run.
"""
THRESHOLD = 0.5

class KEYS:
    '''
    Keys for ResultsCache
    '''
    # ATTRIBUTE_SELECTION = 'attribute selection'
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


class Pipeline:
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
                 ngram=-1,
                 device=torch.device("cuda" if torch.cuda.is_available()
                                     else "cpu"),
                 staggered=True):
        # NOTE Setting any of these to -1 will use the ground truth value
        # self.attr_selection = attr_selection
        self.select_count = select_count
        self.select_column = select_column
        self.predicate_count = predicate_count
        self.predicate_column = predicate_column
        self.predicate_operator = predicate_operator
        self.value_matcher = value_matcher
        self._loaded_model_version = None
        self.device = device
        self.staggered = staggered
        self.ngram = ngram
        self.results = ResultsCache()

    def _delete_models(self):
        # No model is actually loaded - we just store the model numbers and load as needed
        if self._loaded_model_version is None:
            return
        if not isinstance(self.select_count, int):
            del self.select_count
            self.select_count = self._loaded_model_version
        if not isinstance(self.select_column, int):
            del self.select_column
            self.select_column = self._loaded_model_version
        if not isinstance(self.predicate_count, int):
            del self.predicate_count
            self.predicate_count = self._loaded_model_version
        if not isinstance(self.predicate_column, int):
            del self.predicate_column
            self.predicate_column = self._loaded_model_version
        if not isinstance(self.predicate_operator, int):
            del self.predicate_operator
            self.predicate_operator = self._loaded_model_version
        self._loaded_model_version = None

    def _load_select_count(self):
        self._delete_models()
        if self.select_count != -1:
            self._loaded_model_version = self.select_count
            model_name = 'select_count'
            ckpt_fname = f'tb_logs/select-count/version_{self.select_count}/checkpoints/best_model.ckpt'
            self.select_count = load_checkpoint(model_name, ckpt_fname)
            self.select_count = self.select_count.to(self.device).eval()

    def _load_select_column(self):
        self._delete_models()
        if self.select_column != -1:
            self._loaded_model_version = self.select_column
            model_name = 'select_col'
            ckpt_fname = f'tb_logs/select-column/version_{self.select_column}/checkpoints/best_model.ckpt'
            self.select_column = load_checkpoint(model_name, ckpt_fname)
            self.select_column = self.select_column.to(self.device).eval()
    
    def _load_predicate_count(self):
        self._delete_models()
        if self.predicate_count != -1:
            self._loaded_model_version = self.predicate_count
            model_name = 'where_count'
            ckpt_fname = f'tb_logs/predicate-count/version_{self.predicate_count}/checkpoints/best_model.ckpt'
            self.predicate_count = load_checkpoint(model_name, ckpt_fname)
            self.predicate_count = self.predicate_count.to(self.device).eval()

    def _load_predicate_column(self):
        self._delete_models()
        if self.predicate_column != -1:
            self._loaded_model_version = self.predicate_column
            model_name = 'where_col'
            ckpt_fname = f'tb_logs/predicate-column/version_{self.predicate_column}/checkpoints/best_model.ckpt'
            self.predicate_column = load_checkpoint(
                model_name, ckpt_fname).to(self.device).eval()

    def _load_predicate_operator(self):
        self._delete_models()
        if self.predicate_operator != -1:
            self._loaded_model_version = self.predicate_operator
            model_name = 'where_operator'
            ckpt_fname = f'tb_logs/predicate-operator/version_{self.predicate_operator}/checkpoints/best_model.ckpt'
            self.predicate_operator = load_checkpoint(
                model_name, ckpt_fname).to(self.device).eval()

    def _load_value_matcher(self):
        self._delete_models()
        if self.value_matcher != -1:
            self._loaded_model_version = self.value_matcher
            self.value_matcher = TemporalValueMatching(
                table_md_location=cfg.TABLE_METADATA_FP, grams=self.ngram)

    def run_select_count(self, loader):
        print('Loading select count prediction network...')
        self._load_select_count()
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
                    model_out = self.select_count(batch)
                    model_out = torch.argmax(model_out, dim=1).cpu().tolist()
                    self.results.write(model_out)
        print('Done select count prediction!')

    def run_select_column(self, loader):
        print('Loading select column prediction network...')
        self._load_select_column()
        self.results.set_key(KEYS.SELECT_COL)
        number_correct = 0
        number_total = 0
        print('Done loading. Running inference...')
        for batch in tqdm(loader):
            if self.select_column == -1:
                trgs = batch[2]['select']
                for trg in trgs:
                    print(trg)
                    self.results.write_singular([t for t in trg.tolist()])
            else:
                with torch.no_grad():
                    batch = batch_to_device(batch, self.device)
                    model_out = self.select_column(batch)
                    trgs = batch[2]['select']
                    correctness_vec = self.select_column.accuracy_internal(
                        model_out, batch)
                    number_total += correctness_vec.shape[0]
                    number_correct += correctness_vec.sum()

                    model_out = torch.topk(
                        model_out, largest=True, sorted=True, k=6).indices.cpu().tolist()

                    self.results.write(model_out)

        # use the count value to do selection
        for x in self.results.results:
            count = x[KEYS.SELECT_COUNT]
            if count > 0:
                x[KEYS.SELECT_COL] = x[KEYS.SELECT_COL][:count]
            else:
                # Case we don't want to predict anything
                x[KEYS.SELECT_COL] = []
        print('Done predicting select columns!')

    def run_predicate_count(self, loader):
        print('Loading predicate count prediction network...')
        self._load_predicate_count()
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
        self._load_predicate_column()
        self.results.set_key(KEYS.PREDICATE_COL)
        number_correct = 0
        number_total = 0
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
                    trgs = batch[2]['where']
                    correctness_vec = self.predicate_column.accuracy_internal(
                        model_out, batch)
                    number_total += correctness_vec.shape[0]
                    number_correct += correctness_vec.sum()
                    model_out = torch.topk(
                        model_out, largest=True, sorted=True, k=5).indices.cpu().tolist()

                    self.results.write(model_out)
        print('Done predicting predicate columns!')

    def run_predicate_operator(self, loader: DataLoader):
        print('Loading predicate operator prediction network...')
        self._load_predicate_operator()
        self.results.set_key(KEYS.PREDICATE_OP)
        print('Done loading. Finalizing predicate column predictions for input')
        for x in self.results.results:
            count = x[KEYS.PREDICATE_COUNT]
            if count > 0:
                x[KEYS.PREDICATE_COL] = x[KEYS.PREDICATE_COL][:count]
            else:
                # Case we don't want to predict anything
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
                            data_pt_predictions.append(model_out[index])
                            index += 1
                        self.results.write_singular(data_pt_predictions)
        print('Done predicting operators!')

    def run_predicate_value(self, dataset):
        print('Loading value matcher and relevant databases...')
        self._load_value_matcher()
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
                db_id = pt['DEBUG']['db_id']
                predicted_values = []
                for col_index in result_so_far[KEYS.PREDICATE_COL]:
                    try:
                        predicted_values.append(self.value_matcher(
                            db_id, utterance, context, columns[col_index]))
                    except Exception:
                        print(columns, col_index, len(columns))
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
        # print(pred, gt)
        return False


def compute_accuracy(results: ResultsCache, dataset, relative=True, split=CoVis.train_split, context=2, in_part=""):
    '''
    Given a ResultsCache, computes metrics relative or absolute to the ground truth dataset.
    '''


    correctness_vector = []
    stages_false = [0, 0, 0, 0, 0, 0]
    predicted_counts = [0, 0, 0, 0, 0, 0]

    list_context_length = [0 for i in range(context + 1)]
    list_context_length_correct = [0 for i in range(context + 1)]


    # There is a problem here: if a previous stage is wrong, the computation will not arrive the next stage.
    for result, ground_truth in tqdm(zip(results.results, dataset), total=len(dataset)):
        correctness = True 

        if in_part != "predicate":

            if (len(ground_truth['previous_utterances'])) > (context + 1):
                list_context_length[context] += 1
            else:
                list_context_length[len(ground_truth['previous_utterances']) - 1] += 1
            
            if result[KEYS.SELECT_COUNT] != len(ground_truth['select']):
                correctness = False
                stages_false[0] += 1
                if relative:
                    correctness_vector.append(False)
                    continue

            # Selected Columns
            predicted_select_columns = set(result[KEYS.SELECT_COL])

            if predicted_select_columns != set(ground_truth['select']):
                predicted_select_column_names = []
                ground_truth_column_names = []
                for col_idx in predicted_select_columns:
                    predicted_select_column_names.append(ground_truth['columns'][col_idx])
                for col_idx in ground_truth['select']:
                    ground_truth_column_names.append(ground_truth['columns'][col_idx])

                correctness = False
                stages_false[1] += 1

                if relative:
                    correctness_vector.append(False)
                    continue

        if in_part != "select":

            if result[KEYS.PREDICATE_COUNT] != len(ground_truth['where']):
                predicted_counts[result[KEYS.PREDICATE_COUNT]] += 1
                correctness = False
                stages_false[2] += 1

                if relative:
                    correctness_vector.append(False)
                    continue

            # Column of predicates
            predicted_predicate_columns = set(result[KEYS.PREDICATE_COL])
            ground_truth_predicate_columns = set(
                [c[0] for c in ground_truth['where']])

            
            if predicted_predicate_columns != ground_truth_predicate_columns:
                correctness = False
                stages_false[3] += 1

                if relative:
                    correctness_vector.append(False)
                    continue

            # Operator of predicates
            predicted_operators = result[KEYS.PREDICATE_OP]
            gt_encoded = PredicateOperationEncodeTransform(ground_truth['where'])
            ground_truth_predicate_operators = [c[1]
                                                for c in gt_encoded]
                    
            if predicted_operators != ground_truth_predicate_operators:
                correctness = False
                stages_false[4] += 1
                if relative:
                    correctness_vector.append(False)
                    continue

            # Value of predicates
            gt_values = [c[-1] for c in gt_encoded]
            predicted_values = result[KEYS.PREDICATE_VALUE]
            all_values_correct = True
            for p, g in zip(predicted_values, gt_values):
                all_values_correct = all_values_correct and compare_values(p, g)
            if not all_values_correct:
                correctness = False
                stages_false[5] += 1
                if relative:
                    correctness_vector.append(False)
                    continue
        
        # Making here means the two are the same
        correctness_vector.append(correctness)

        if (len(ground_truth['previous_utterances'])) > (context + 1):
            list_context_length_correct[context] += 1
        else:
            list_context_length_correct[len(ground_truth['previous_utterances']) - 1] += 1

    return torch.Tensor(correctness_vector), stages_false, predicted_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=['train', 'val', 'test'], type=str)
    parser.add_argument('--select_col', default=0, type=int)
    parser.add_argument('--select_count', default=0, type=int)
    parser.add_argument('--where_col', default=0, type=int)
    parser.add_argument('--where_count', default=0, type=int)
    parser.add_argument('--where_operator', default=0, type=int)
    parser.add_argument('--where_value', default=0, type=int)
    parser.add_argument('--context', type=int, default=1)
    parser.add_argument('--verbosity', type=int, default=1)
    parser.add_argument('--no_relative', action='store_true',
                        help='computes metrics without relative accuracy. Relative means downstream errors are not accounted for twice.')
    parser.add_argument('--ngram', type=int, default=-1)
    parser.add_argument('--in_part', choices=['predicate', 'select', "all"], type=str, default="all")
    args = parser.parse_args()

    if args.split == 'train':
        split = CoVis.train_split
    elif args.split == 'test':
        split = CoVis.test_split
    else:
        split = CoVis.dev_split

    model = Pipeline(select_count=args.select_count, select_column=args.select_col, predicate_count=args.where_count, predicate_column=args.where_col, predicate_operator=args.where_operator, value_matcher=args.where_value, ngram=args.ngram)

    model_dset = CoVis(split,
                       utterance_transforms=transforms.
                       default_utterance_transforms(),
                       where_transforms=transforms.default_where_transform(),
                       column_transforms=transforms.default_column_transforms(),
                       context_sentences=args.context, oversampling=False)

    model_dloader = DataLoader(
        model_dset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate,
        num_workers=4)

    # and one version for the raw text value matcher
    value_dset = CoVis(split, context_sentences=args.context,
                       seperate_context=True, oversampling=False)

    if args.verbosity < 1:
        with open('eval_out.txt', 'w') as f:
                with redirect_stderr(f):
                    with redirect_stdout(f):
                        # Run select count prediction
                        model.run_select_count(model_dloader)

                        # Run select column prediction
                        model.run_select_column(model_dloader)
                        
                        # Run predicate count prediction
                        model.run_predicate_count(model_dloader)

                        # Run predicate column prediction
                        model.run_predicate_column(model_dloader)

                        # Run predicate operator prediction
                        model.run_predicate_operator(model_dloader)

                        # Run value matching
                        model.run_predicate_value(value_dset)
    else:
        model.run_select_count(model_dloader)

        # Run select column prediction
        model.run_select_column(model_dloader)
        
        # Run predicate count prediction
        model.run_predicate_count(model_dloader)

        # Run predicate column prediction
        model.run_predicate_column(model_dloader)

        # Run predicate operator prediction
        model.run_predicate_operator(model_dloader)

        # Run value matching
        model.run_predicate_value(value_dset)
    

    # Test Accuracy
    correctness_vector, error_stages, predicted_counts = compute_accuracy(
        model.results, value_dset, relative=not args.no_relative, split=args.split, in_part=args.in_part)
    print('====== DONE ======')
    print(f'Total number of data points: {correctness_vector.shape[0]}')
    num_incorrect = correctness_vector.shape[0] - \
        correctness_vector.sum().item()
    
    print(
        f'Number of total correct data points: {correctness_vector.sum().item()} and Incorrect: {correctness_vector.shape[0] - correctness_vector.sum().item()}')
    print(
        f'Overall Accuracy: {(correctness_vector.sum().item() / correctness_vector.shape[0])*100:.4f}%')

    model.results.dump(f'{split}.pkl')
