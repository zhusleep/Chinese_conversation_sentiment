#-*-coding:utf-8-*-
import pickle, torch
from spo_model import SPOModel
import torch.nn as nn


class SentiAna(object):

    def __init__(self,model_path):
        self.model_path = model_path
        self.model = SPOModel(vocab_size=22161,
                 word_embed_size=200, encoder_size=128, dropout=0.5,
                 seq_dropout=0.0, init_embedding=None,
                 dim_num_feat=1)
        self.cpu = not torch.cuda.is_available()
        self.load_model()
        self.load_tokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_tokenizer(self):
        self.t = pickle.load(open('tokenizer.pkl', 'rb'))

    def load_model(self):
        state = torch.load(self.model_path)
        self.model.load_state_dict(state['state_dict'], strict=True)

    def predict(self,text):
        """
        预测文本的情感状态，１表示positive,0表示negative
        :param text: ［你，真，厉害］　已经被分词
        :return:
        """
        text = text[0:25]
        max_length = len(text)
        numerical_features = [[len(text)]]
        length = max_length
        X = self.t.transform([text])
        X = torch.tensor(X)
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.to(self.device)
        numerical_features = torch.tensor(numerical_features)

        n_feats = numerical_features.type(torch.float).to(self.device)
        mask_X = torch.ones(1, max_length, dtype=torch.uint8).type(torch.float).to(self.device)
        length = torch.tensor([length]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X, mask_X, length, n_feats)
        pred = pred.cpu().numpy()[0, 0]
        return pred

SA = SentiAna('model/senti.pth')
print(SA.predict(['谢谢','你']))
