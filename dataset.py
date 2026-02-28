import os
import torch
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class TextDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42

    def __init__(self, de_file: str, en_file: str = None, sp_model_prefix: str = None,
                 vocab_size: int = 2000, normalization_rule_name: str = 'nmt_nfkc_cf',
                 model_type: str = 'bpe', max_length: int = 128, train_ratio=1.0):
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
        if not os.path.isfile(sp_model_prefix + '_de.model'):
            # train tokenizer if not trained yet
            SentencePieceTrainer.train(
                input=de_file, vocab_size=vocab_size,
                model_type=model_type, model_prefix=f"{sp_model_prefix}_de",
                normalization_rule_name=normalization_rule_name,
                pad_id=0, bos_id=1, eos_id=2, unk_id=3
            )

        # load tokenizer from file
        self.sp_models = [SentencePieceProcessor(model_file=sp_model_prefix + '_de.model')]

        self.texts = [open(de_file, "r").readlines()]
        n_texts = len(self.texts[0])

        if self.train:
            if not os.path.isfile(sp_model_prefix + '_en.model'):
                # train tokenizer if not trained yet
                SentencePieceTrainer.train(
                    input=en_file, vocab_size=vocab_size,
                    model_type=model_type, model_prefix=f"{sp_model_prefix}_en",
                    normalization_rule_name=normalization_rule_name,
                    pad_id=0, bos_id=1, eos_id=2, unk_id=3
                )
            self.sp_models.append(SentencePieceProcessor(model_file=sp_model_prefix + '_en.model'))
            self.texts.append(open(en_file, "r").readlines())

        self.texts = [texts[:int(n_texts * train_ratio)] for texts in self.texts]


        self.indices = [model.encode(texts) for model, texts in zip(self.sp_models, self.texts)]

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.sp_models[0].pad_id(), self.sp_models[0].unk_id(), \
            self.sp_models[0].bos_id(), self.sp_models[0].eos_id()
        self.max_length = max_length
        self.vocab_size = self.sp_models[0].vocab_size()

    def text2ids(self, texts: Union[str, List[str]], lang: str = 'de') -> Union[List[int], List[List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        model_id = 0 if lang == 'de' else 1
        return self.sp_models[model_id].encode(texts)

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
        return self.sp_models[model_id].decode(ids)

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