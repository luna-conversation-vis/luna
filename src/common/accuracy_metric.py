from random import sample
from src.common.utils import print_dict


class AccuracyMetric:
    """
    Helper class that stores accuracy. The class could be generalized to other metrics but is not tested.
    """

    def __init__(self, name):
        self.name = name
        self._correct = 0
        self._total = 0
        self._examples = []

    def correct(self):
        self._correct += 1

    def total(self):
        self._total += 1

    def compute(self):
        return self._correct / self._total

    def store_example(self, **kwargs):
        self._examples.append(kwargs)

    def __repr__(self):
        return f'{self.name}: {self.compute() * 100:.2f}%' + f' Subcount: {self._correct} Total: {self._total}'

    def sample_examples(self, k):
        if k > len(self._examples):
            k = len(self._examples)
        return sample(self._examples, k)

    def print_examples(self, k):
        examples_to_print = self.sample_examples(k=k)
        for i, example in enumerate(examples_to_print):
            print(f'EXAMPLE {i+1}')
            print_dict(example)

    @property
    def examples(self):
        return self._examples
