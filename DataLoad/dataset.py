import torch
import torchtext
from torchtext import data

def tokenizer(comment):
    return comment.split()

def get_dataset(data_path, vec, tokenizer1=tokenizer):
    # just the whitespace split function
    TEXT = data.Field(tokenize=tokenizer1, lower=True, sequential=True, fix_length=200)
    LABELS = data.Field(sequential=False)
    INDEX = data.Field(sequential=False)
    gold_train, val, test = data.TabularDataset.splits(
        path=data_path, format="csv",train='train.csv', test='test.csv', validation='val.csv',
        skip_header=True,
        fields=[('text', TEXT), ('label', LABELS), ('id', INDEX)])


    gold_train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (gold_train, val, test), batch_sizes=(1024, 256, 256),
        sort_within_batch=True,
        sort_key=lambda x: len(x.text))



    TEXT.build_vocab(gold_train, val, test, min_freq=20, vectors=vec)
    LABELS.build_vocab(gold_train)
    return gold_train_iter, val_iter, test_iter, TEXT, LABELS, INDEX

