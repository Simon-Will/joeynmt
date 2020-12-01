# coding: utf-8
"""
Data module
"""
import itertools
import sys
import random
import os
import os.path
from typing import List, Optional
import logging

from torchtext.datasets import TranslationDataset
from torchtext import data
from torchtext.data import Batch, Dataset, Iterator, Field
import torch

from joeynmt.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN
from joeynmt.vocabulary import build_vocab, Vocabulary

logger = logging.getLogger(__name__)


def load_data(data_cfg: dict, datasets: list = None)\
        -> (Dataset, Optional[Dataset], Dataset, Optional[Dataset],
            Optional[Dataset], Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :param datasets: list of dataset names to load
    :return:
        - train_data: training dataset
        - train2_data: second training dataset if given, otherwise None
        - dev_data: development dataset
        - dev2_data: second development dataset if given, otherwise None
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    if datasets is None:
        datasets = ["train", "train2", "dev", "dev2", "test"]

    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg.get("train", None)
    train2_path = data_cfg.get("train2", None)
    dev_path = data_cfg.get("dev", None)
    dev2_path = data_cfg.get("dev2", None)
    test_path = data_cfg.get("test", None)

    if train_path is None and dev_path is None and test_path is None:
        raise ValueError('Please specify at least one data source path.')

    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]
    max_sent_length = data_cfg["max_sent_length"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    trg_field = data.Field(init_token=BOS_TOKEN, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           unk_token=UNK_TOKEN,
                           batch_first=True, lower=lowercase,
                           include_lengths=True)

    train_data = None
    if "train" in datasets and train_path is not None:
        logger.info("loading training data...")
        train_data = TranslationDataset(path=train_path,
                                        exts=("." + src_lang, "." + trg_lang),
                                        fields=(src_field, trg_field),
                                        filter_pred=
                                        lambda x: len(vars(x)['src'])
                                        <= max_sent_length
                                        and len(vars(x)['trg'])
                                        <= max_sent_length)

        random_train_subset = data_cfg.get("random_train_subset", -1)
        if random_train_subset > -1:
            # select this many training examples randomly and discard the rest
            keep_ratio = random_train_subset / len(train_data)
            keep, _ = train_data.split(
                split_ratio=[keep_ratio, 1 - keep_ratio],
                random_state=random.getstate())
            train_data = keep

    train2_data = None
    if "train2" in datasets and train2_path is not None:
        logger.info("loading training 2 data...")
        train2_data = TranslationDataset(path=train2_path,
                                         exts=("." + src_lang, "." + trg_lang),
                                         fields=(src_field, trg_field),
                                         filter_pred=
                                         lambda x: len(vars(x)['src'])
                                         <= max_sent_length
                                         and len(vars(x)['trg'])
                                         <= max_sent_length)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)

    assert (train_data is not None) or (src_vocab_file is not None)
    assert (train_data is not None) or (trg_vocab_file is not None)

    logger.info("building vocabulary...")
    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)
    trg_vocab = build_vocab(field="trg", min_freq=trg_min_freq,
                            max_size=trg_max_size,
                            dataset=train_data, vocab_file=trg_vocab_file)

    dev_data = None
    if "dev" in datasets and dev_path is not None:
        logger.info("loading dev data...")
        dev_data = TranslationDataset(path=dev_path,
                                      exts=("." + src_lang, "." + trg_lang),
                                      fields=(src_field, trg_field))

    dev2_data = None
    if "dev2" in datasets and dev2_path is not None:
        logger.info("loading dev 2 data...")
        dev_data = TranslationDataset(path=dev2_path,
                                      exts=("." + src_lang, "." + trg_lang),
                                      fields=(src_field, trg_field))

    test_data = None
    if "test" in datasets and test_path is not None:
        logger.info("loading test data...")
        # check if target exists
        if os.path.isfile(test_path + "." + trg_lang):
            test_data = TranslationDataset(
                path=test_path, exts=("." + src_lang, "." + trg_lang),
                fields=(src_field, trg_field))
        else:
            # no target is given -> create dataset from src only
            test_data = MonoDataset(path=test_path, ext="." + src_lang,
                                    field=src_field)
    src_field.vocab = src_vocab
    trg_field.vocab = trg_vocab
    logger.info("data loaded.")
    return (train_data, train2_data, dev_data, dev2_data, test_data, src_vocab,
            trg_vocab)


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


# pylint: disable=unused-argument,global-variable-undefined
def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


class MultiBatchIterator:

    def __init__(self, iterators: List[Iterator]):
        self.iterators = iterators

    def __iter__(self):
        for batches in zip(*self.iterators):
            dataset = batches[0].dataset
            merged_data = {}
            for field_name, field in dataset.fields.items():
                tensors = []
                lengths = []
                for batch in batches:
                    batch_data = getattr(batch, field_name)
                    tensors.append(batch_data[0])
                    lengths.append(batch_data[1])
                tensor_lengths = [t.size()[1] for t in tensors]
                idx_of_longest, max_len = max(
                    zip(range(len(tensors)), tensor_lengths),
                    key=lambda idx_and_length: idx_and_length[1]
                )
                for i, tensor in enumerate(tensors):
                    if i == idx_of_longest:
                        continue
                    size = tensor.size()
                    tensors[i] = torch.cat(
                        [tensor,
                         torch.ones(size[0], max_len - size[1],
                                    dtype=tensor.dtype)
                         * field.vocab.stoi[field.pad_token]],
                        1
                    )
                merged_data[field_name] = (torch.cat(tensors, 0),
                                           torch.cat(lengths, 0))
                # TODO: Sort if dataset.sort_within_batch

            batch_size = sum(batch.batch_size for batch in batches)
            yield Batch.fromvars(batches[0].dataset, batch_size, **merged_data)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   dataset2: Dataset = None,
                   batch_type: str = "sentence",
                   dataset2_ratio: float = 0.5,
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """
    assert 0 <= dataset2_ratio <= 1

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if dataset2:
        batch2_size = round(dataset2_ratio * batch_size)
        batch_size -= batch2_size

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
        if dataset2:
            data2_iter = data.BucketIterator(
                repeat=False, sort=False, dataset=dataset2,
                batch_size=batch2_size, batch_size_fn=batch_size_fn,
                train=True, sort_within_batch=True,
                sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=False, sort=False)
        if dataset2:
            data2_iter = data.BucketIterator(
                repeat=False, dataset=dataset2,
                batch_size=batch2_size, batch_size_fn=batch_size_fn,
                train=False, sort=False)

    if dataset2:
        data_iter = MultiBatchIterator([data_iter, data2_iter])

    return data_iter


class MonoDataset(Dataset):
    """Defines a dataset for machine translation without targets."""

    @staticmethod
    def sort_key(ex):
        return len(ex.src)

    def __init__(self, path: str, ext: str, field: Field, **kwargs) -> None:
        """
        Create a monolingual dataset (=only sources) given path and field.

        :param path: Prefix of path to the data file
        :param ext: Containing the extension to path for this language.
        :param field: Containing the fields that will be used for data.
        :param kwargs: Passed to the constructor of data.Dataset.
        """

        fields = [('src', field)]

        if hasattr(path, "readline"):  # special usage: stdin
            src_file = path
        else:
            src_path = os.path.expanduser(path + ext)
            src_file = open(src_path)

        examples = []
        for src_line in src_file:
            src_line = src_line.strip()
            if src_line != '':
                examples.append(data.Example.fromlist(
                    [src_line], fields))

        src_file.close()

        super().__init__(examples, fields, **kwargs)
