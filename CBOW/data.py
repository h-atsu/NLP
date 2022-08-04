import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import create_contexts_target
from dezero.datasets import Dataset
from dezero.dataloaders import DataLoader
from datasets.ptb import load_data


class PTB(Dataset):
    def __init__(self, data_type='train', window_size=1):
        '''
        :param data_type: データの種類 : 'train' or 'test' or 'valid (val)'        
        '''
        self.data_type = data_type
        self.window_size = window_size
        super().__init__()

    def prepare(self):
        self.corpus, self.word_to_id, self.id_to_word = load_data(
            self.data_type)

        contexts, target = create_contexts_target(
            self.corpus, self.window_size)
        self.data = contexts
        self.label = target


class SeqPTB(Dataset):
    def __init__(self, data_type='train', corpus_size=1000):
        self.data_type = data_type
        self.corpus_size = corpus_size
        super().__init__()

    def prepare(self):
        self.corpus, self.word_to_id, self.id_to_word = load_data(
            self.data_type)
        self.corpus = self.corpus[:self.corpus_size]
        self.vocab_size = int(max(self.corpus) + 1)
        self.data = self.corpus[:-1]
        self.label = self.corpus[1:]


if __name__ == '__main__':
    dataset = PTB('train', window_size=3)
    print(dataset)
