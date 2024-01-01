from enum import Enum
from os.path import join
from .models.builder import add_subparsers
from .models.builder import _models

"""Configuration file. Use 'import cfg from src.config' to use."""


class CONFIG:
    '''constants'''
    TRAIN_FNAME = 'train_multi.json'
    DEV_FNAME = 'dev_multi.json'
    TEST_FNAME = 'test_multi.json'
    DATA_ROOT = 'data/'
    COVIS_ROOT = join(DATA_ROOT, 'covis/')
    DATA_TABLES = join(DATA_ROOT, 'cosql_dataset/database/')
    TRAIN_BATCH_SIZE = 3
    TEST_BATCH_SIZE = 24
    MODEL_CHECKPOINTS = './checkpoints/'
    TABLE_METADATA_FP = join(DATA_ROOT, 'cosql_dataset/tables.json')


cfg = CONFIG()


class WhereCategories(str, Enum):
    NO_CHANGE = 'No Change'
    GENERALIZE = 'Generalize'
    ADD = 'Add'
    PIVOT_SINGULAR = 'Pivot Singular'
    PIVOT_MULTIPLE = 'Pivot Multiple'
    FAILED = 'Failed'


def setup_training_argparse(parser):
    '''Args for training - different from PyTorch lightnings args'''
    parser.add_argument('--context_sentences', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_dataloading_workers', type=int, default=4)
    parser.add_argument('--save_using_acc', action='store_true')
    subparser = parser.add_subparsers(dest='model_name')
    add_subparsers(subparser)
    return parser


def setup_inference_argparse(parser):
    parser.add_argument('model_name', choices=list(_models.keys()))
    parser.add_argument('ckpt', type=str)
    parser.add_argument('--context_sentences', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=30)
    parser.add_argument('--num_dataloading_workers', type=int, default=4)
    parser.add_argument('--ngpu', type=int, default=1)
    return parser


# Easy way to set pytorch lightning default config
trainer_args = {
    'devices': 1,
    'accelerator': 'gpu',
    'max_epochs': 8,
}
