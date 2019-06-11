#-*-coding:utf-8-*-
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm as tqdm


class SPO(Dataset):
    def __init__(self, X, tokenizer, pos=None, pos_t=None, max_len=25, label=None,  ner=None, combined_char_t=None):
        super(SPO, self).__init__()
        self.max_len = max_len
        X = self.limit_length(X)
        self.raw_X = X
        self.raw_pos = pos
        self.label = label
        self.tokenizer = tokenizer
        self.pos_t = pos_t
        self.X = tokenizer.transform(X)
        # X = pad_sequences(X, maxlen=198)
        self.length = [len(sen) for sen in self.X]
        self.numerical_df = self.cal_numerical_f(self.X)

    def limit_length(self, X):
        temp = []
        for item in X:
            temp.append(item[0:self.max_len])
        return temp

    def cal_word_char_token(self):
        sentence = []
        for item in self.raw_X:
            sentence.append(self.combined_char_t.transform(item))
        return sentence

    def cal_numerical_f(self, arr):
        length = [len(sen) for sen in arr]
        length = np.array(length).reshape((-1, 1))

        return length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        numerical_features = self.numerical_df[index]
        if self.label is not None:
            label = self.label[index]
        length = self.length[index]
        # if self.ner is not None:
        #     ner = torch.tensor(self.ner[index])
        #if self.combined_char_t is not None:

        if self.label is not None:
            return index, sentence, length, numerical_features, label
        else:
            return index, sentence, length, numerical_features




class SPO_BERT(Dataset):
    def __init__(self, X, pos, tokenizer, pos_t, label=None,  ner=None, combined_char_t=None):
        super(SPO_BERT, self).__init__()
        self.raw_X = X
        self.raw_pos = pos
        self.label = label
        self.tokenizer = tokenizer
        self.pos_t = pos_t
        self.X = self.deal_for_bert(X, tokenizer)
        self.pos = pos_t.transform(pos)
        # X = pad_sequences(X, maxlen=198)
        self.length = [len(sen) for sen in self.X]
        self.numerical_df = self.cal_numerical_f(self.X)
        self.ner = ner
        self.combined_char_t = combined_char_t
        if combined_char_t is not None:
            self.combined_char = self.cal_word_char_token()

    def deal_for_bert(self,x,t):
        temp = []
        for item in tqdm(x):
            sen = t.tokenize(''.join(item))
            sen = ['[CLS]']+sen+['[SEP]']
            indexed_tokens = t.convert_tokens_to_ids(sen)
            temp.append(indexed_tokens)
        return temp

    def cal_word_char_token(self):
        sentence = []
        for item in self.raw_X:
            sentence.append(self.combined_char_t.transform(item))
        return sentence

    def cal_numerical_f(self, arr):
        length = [len(sen) for sen in arr]
        length = np.array(length).reshape((-1, 1))

        return length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        pos = torch.tensor(self.pos[index])
        if len(sentence)!=len(pos):
            i=1
        numerical_features = self.numerical_df[index]
        label = self.label[index]
        length = self.length[index]
        # if self.ner is not None:
        #     ner = torch.tensor(self.ner[index])
        #if self.combined_char_t is not None:

        if label is not None:
            if self.combined_char_t is not None:
                char_vocab = self.combined_char[index]
                return index, sentence, pos, length, numerical_features, char_vocab, label
            else:
                return index, sentence, pos, length, numerical_features, label
        else:
            return index, sentence, pos, length, numerical_features


class SPO_NER(Dataset):
    def __init__(self, X, pos, tokenizer, pos_t, ner_t=None, max_len=300, label=None,  ner=None, combined_char_t=None):
        super(SPO_NER, self).__init__()
        self.max_len = max_len
        X = self.limit_length(X)
        pos = self.limit_length(pos)
        self.raw_X = X
        self.raw_pos = pos
        if ner_t is not None:
            self.label = ner_t.transform(label)
        else:
            self.label = label
        self.tokenizer = tokenizer
        self.pos_t = pos_t
        self.X = tokenizer.transform(X)
        self.pos = pos_t.transform(pos)
        # X = pad_sequences(X, maxlen=198)
        self.length = [len(sen) for sen in self.X]
        self.numerical_df = self.cal_numerical_f(self.X)
        self.ner = ner
        self.combined_char_t = combined_char_t
        if combined_char_t is not None:
            self.combined_char = self.cal_word_char_token()
        # test

    def limit_length(self, X):
        temp = []
        for item in X:
            temp.append(item[0:self.max_len])
        return temp

    def cal_word_char_token(self):
        sentence = []
        for item in self.raw_X:
            sentence.append(self.combined_char_t.transform(item))
        return sentence

    def cal_numerical_f(self, arr):
        length = [len(sen) for sen in arr]
        length = np.array(length).reshape((-1, 1))

        return length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        pos = torch.tensor(self.pos[index])
        if len(sentence)!=len(pos):
            i=1
        numerical_features = self.numerical_df[index]
        label = self.label[index]
        length = self.length[index]
        if self.ner is not None:
            ner = torch.tensor(self.ner[index])
        #if self.combined_char_t is not None:

        if label is not None:
            if self.combined_char_t is not None:
                char_vocab = self.combined_char[index]
                return index, sentence, pos, length, numerical_features, char_vocab, label
            elif self.ner is not None:
                return index, sentence, pos, length, numerical_features, ner, label
            else:
                return index, sentence, pos, length, numerical_features, label
        else:
            return index, sentence, pos, length, numerical_features


def pad_sequence(sequences):
    max_len = max([len(s) for s in sequences])
    out_mat = [[torch.tensor([0])]*max_len]*len(sequences)
    for i, sen in enumerate(sequences):
        sen = [torch.tensor(w) for w in sen]
        length = len(sen)
        out_mat[i][:length] = sen

    return out_mat


def collate_fn(batch):

    if len(batch[0]) == 5:
        index, X, length, numerical_feats, label = zip(*batch)
        length = torch.tensor(length, dtype=torch.int)
        numerical_feats = torch.tensor(numerical_feats)
        label = torch.tensor(label)
        return index, X, length, numerical_feats, label
    else:
        index, X, length, numerical_feats = zip(*batch)
        length = torch.tensor(length, dtype=torch.int)
        numerical_feats = torch.tensor(numerical_feats)
        return index, X, length, numerical_feats


def get_mask(sequences_batch, sequences_lengths, is_cuda=True):
    """
    Get the mask for a batch of padded variable length sequences.
    Args:
        sequences_batch: A batch of padded variable length sequences
            containing word indices. Must be a 2-dimensional tensor of size
            (batch, sequence).
        sequences_lengths: A tensor containing the lengths of the sequences in
            'sequences_batch'. Must be of size (batch).
    Returns:
        A mask of size (batch, max_sequence_length), where max_sequence_length
        is the length of the longest sequence in the batch.
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.uint8)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    if is_cuda:
        return mask.cuda()
    else:
        return mask


