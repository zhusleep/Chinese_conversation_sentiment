#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
import torch, os
from spo_dataset import SPO, get_mask, collate_fn
from spo_model import SPOModel
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tokenize_pkg.tokenize import Tokenizer
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, load_glove, get_threshold
import logging
import time
from sklearn.metrics import classification_report

def read_data(path):
    data = []
    label = []
    with open(path,'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            text = line[1].split()
            if not text:continue
            if line[0] =='positive':
                label.append(1)
                data.append(text)
            elif line[0] == 'negative':
                label.append(0)
                data.append(text)
            else:
                continue
    data_df = pd.DataFrame()
    data_df['text'] = data
    data_df['label'] = label
    return data_df


train_data = read_data('data/sentiment_XS_30k.txt')
valid_data = read_data('data/sentiment_XS_test.txt')
print(train_data.sample(100))


current_name = 'log/%s.txt' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.basicConfig(filename=current_name,
                    filemode='w',
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


seed_torch(2019)

t = Tokenizer(max_feature=500000, segment=False)
t.fit(list(train_data['text'].values) + list(valid_data['text'].values))

print('一共有%d 个词' % t.num_words)

train_dataset = SPO(train_data['text'].values, t, label=train_data['label'])
valid_dataset = SPO(valid_data['text'].values, t, label=valid_data['label'])

batch_size = 40


# 准备embedding数据
#embedding_file = 'embedding/miniembedding_engineer_baike_word.npy'
embedding_file = 'embedding/miniembedding_engineer_qq_att.npy'

if os.path.exists(embedding_file):
    embedding_matrix = np.load(embedding_file)
else:
    #embedding = '/home/zhukaihua/Desktop/nlp/embedding/baike'
    embedding = '/home/zhukaihua/Desktop/nlp/embedding/Tencent_AILab_ChineseEmbedding.txt'
    embedding_matrix = load_glove(embedding, t.num_words+100, t)
    np.save(embedding_file, embedding_matrix)

model = SPOModel(vocab_size=embedding_matrix.shape[0],
                 word_embed_size=embedding_matrix.shape[1], encoder_size=128, dropout=0.5,
                 seq_dropout=0.0, init_embedding=embedding_matrix,
                 dim_num_feat=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=batch_size)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, shuffle=False, batch_size=batch_size)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
clip = 50

for epoch in range(25):
    model.train()
    train_loss = 0
    for index, X, length, numerical_features, label in tqdm(train_dataloader):
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.cuda()
        #length = length.cuda()

        n_feats = numerical_features.type(torch.float).cuda()
        label = label.type(torch.float).cuda()
        mask_X = get_mask(X, length, is_cuda=True).type(torch.float)
        pred = model(X, mask_X, length, n_feats)

        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients: gradients are modified in place
        _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
        #_ = nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_loss += loss.item()
        #break
    train_loss = train_loss/len(train_data)

    model.eval()
    valid_loss = 0
    pred_set = []
    label_set = []
    for index, X, length, numerical_features, label in tqdm(valid_dataloader):
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.cuda()
        length = length.cuda()
        n_feats = numerical_features.type(torch.float).cuda()
        label = label.type(torch.float).cuda()
        mask_X = get_mask(X, length, is_cuda=True).type(torch.float)
        with torch.no_grad():
            pred = model(X, mask_X, length, n_feats)
        loss = loss_fn(pred, label)
        pred_set.append(pred.cpu().numpy())
        label_set.append(label.cpu().numpy())
        valid_loss += loss
    valid_loss = valid_loss/len(valid_data)
    pred_set = np.concatenate(pred_set, axis=0)
    label_set = np.concatenate(label_set, axis=0)
    print('train loss %f,val loss %f' % (train_loss, valid_loss))
    INFO_THRE,thre_list = get_threshold(pred_set, label_set)
    print(INFO_THRE)