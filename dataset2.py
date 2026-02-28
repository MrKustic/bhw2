import os
import torch
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import itertools


class TextDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42

    def __init__(self, de_file: str, en_file: str = None,
                 vocab_size: int = 2000, max_length: int = 128, train_ratio=1.0):
        """
        Dataset with texts, supporting BPE tokenizer    
        :param data_file: txt file containing texts
        :param train: whether to use train or validation split
        :param sp_model_prefix: path prefix to save tokenizer model
        :param vocab_size: sentencepiece tokenizer vocabulary size
        :param normalization_rule_name: sentencepiece tokenizer normalization rule
        :param model_type: sentencepiece tokenizer model type
        :param max_length: maximal length of text in tokens
        """
        self.train = en_file is not None
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = 0, 3, 1, 2

        self.texts = [open(de_file, "r").readlines()]
        n_texts = len(self.texts[0])

        if self.train:
            self.texts.append(open(en_file, "r").readlines())

        self.texts = [texts[:int(n_texts * train_ratio)] for texts in self.texts]

        self.id2word = []
        self.word2id = []
        self.indices = []
        self.vocab_size = 0

        for id_model, texts in enumerate(self.texts):
            lang = 'en' if id_model == 1 else 'de'
            words = list(itertools.chain.from_iterable([list(text.split()) for text in texts]))
            words = [x for x, y in Counter(words).most_common(vocab_size - 4)]
            self.word2id.append({word: i + 4 for i, word in enumerate(words)})
            self.id2word.append({id: word for word, id in self.word2id[-1].items()} | {0: "", 1: "", 2: "", 3: ""})
            self.indices.append(self.text2ids(texts, lang))
            self.vocab_size = max(self.vocab_size, len(words))

        self.max_length = max_length

    def id2word_(self, id, id_model):
        return self.id2word[id_model].get(id, "??")

    def word2id_(self, word, id_model):
        return self.word2id[id_model].get(word, self.unk_id)

    def text2ids(self, texts: Union[str, List[str]], lang: str = 'de') -> Union[List[int], List[List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        model_id = 0 if lang == 'de' else 1
        if type(texts) == str:
            return [self.word2id_(word, model_id) for word in texts.split()]
        return [[self.word2id_(word, model_id) for word in text.split()] for text in texts]

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]], lang: str = 'de') -> Union[str, List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        model_id = 0 if lang == 'de' else 1
        if type(ids[0]) == int:
            return " ".join([self.id2word_(id, model_id) for id in ids])
        return [" ".join([self.id2word_(id, model_id) for id in id_list]) for id_list in ids]

    def __len__(self):
        """
        Size of the dataset
        :return: number of texts in the dataset
        """
        return len(self.indices[0])

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: encoded text indices and its actual length (including BOS and EOS specials)
        """
        """
        Take corresponding index array from self.indices,
        add special tokens (self.bos_id and self.eos_id) and 
        pad to self.max_length using self.pad_id.
        Return padded indices of size (max_length, ) and its actual length
        """
        result = []
        for indices_list in self.indices:
            length = min(len(indices_list[item]), self.max_length - 2)
            indices = torch.tensor([self.bos_id] + indices_list[item][:length] + [self.eos_id] + [self.pad_id] * (self.max_length - length - 2))
            result.append((indices, length + 2))

        return result
