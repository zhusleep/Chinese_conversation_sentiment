#-*-coding:utf-8-*-
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_pretrained_bert.modeling import BertModel
import math



class LSTMEncoder(nn.Module):
    def __init__(self,
                 embed_size=200,
                 encoder_size=64,
                 bidirectional=True
                 ):
        super(LSTMEncoder, self).__init__()
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=encoder_size,
                               bidirectional=bidirectional,
                               num_layers=1,
                               batch_first=True)

    def forward(self, x, len_x):
        # 对输入的batch按照长度进行排序
        sorted_seq_lengths, indices = torch.sort(len_x, descending=True)
        # 排序前的顺序可通过再次排序恢复
        _, desorted_indices = torch.sort(indices, descending=False)
        x = x[indices]
        packed_inputs = nn.utils.rnn.pack_padded_sequence(x, sorted_seq_lengths, batch_first=True)
        res, state = self.encoder(packed_inputs)
        padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=True)
        res = padded_res[desorted_indices]
        return res


class GRUEncoder(nn.Module):
    def __init__(self,
                 embed_size=200,
                 encoder_size=64,
                 bidirectional=True
                 ):
        super(GRUEncoder, self).__init__()
        self.encoder = nn.GRU(input_size=embed_size,
                               hidden_size=encoder_size,
                               bidirectional=bidirectional,
                               num_layers=1,
                               batch_first=True)

    def forward(self, x, len_x):
        # 对输入的batch按照长度进行排序
        sorted_seq_lengths, indices = torch.sort(len_x, descending=True)
        # 排序前的顺序可通过再次排序恢复
        _, desorted_indices = torch.sort(indices, descending=False)
        x = x[indices]
        packed_inputs = nn.utils.rnn.pack_padded_sequence(x, sorted_seq_lengths, batch_first=True)
        res, state = self.encoder(packed_inputs)
        padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=True)
        res = padded_res[desorted_indices]
        return res


def masked_softmax(x, mask, dim=-1, memory_efficient=False, mask_fill_value=-1e32):
    if mask is None:
        result = F.softmax(x, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = F.softmax(x * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = x.masked_fill((1 - mask).byte(), mask_fill_value)
            result = F.softmax(masked_vector, dim=dim)

    return result


class SoftAttention(nn.Module):
    def forward(self, a, b, mask_a, mask_b):
        attention_matrix = a.bmm(b.transpose(2, 1).contiguous())
        attention_mask = mask_a.bmm(mask_b.transpose(2, 1).contiguous())

        weight_for_a = masked_softmax(attention_matrix, attention_mask, -1)
        attended_a = weight_for_a.bmm(b)

        attention_matrix = attention_matrix.transpose(2, 1).contiguous()
        attention_mask = attention_mask.transpose(2, 1).contiguous()
        weight_for_b = masked_softmax(attention_matrix, attention_mask, -1)
        attended_b = weight_for_b.bmm(a)

        return attended_a, attended_b


# mask attention
class Attention(nn.Module):
    def __init__(self, feature_dim, bias=False, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        # if bias:
        #     self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = x.size()[1]

        # step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


class SPOModel(nn.Module):
    def __init__(self,
                 vocab_size=20000,
                 word_embed_size=300,
                 init_embedding=None,
                 bidirectional=True,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.5,
                 seq_dropout=0.1
                 ):
        super(SPOModel, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size,
                                           word_embed_size,
                                           padding_idx=0)
        self.seq_dropout = seq_dropout
        self.embed_size = word_embed_size
        self.filter_num = 50
        if init_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.LSTM = LSTMEncoder(embed_size=self.embed_size,
                                   encoder_size=encoder_size,
                                   bidirectional=bidirectional)
        self.GRU = GRUEncoder(embed_size=self.embed_size,
                                encoder_size=encoder_size,
                                bidirectional=bidirectional)
        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        self.attention = SoftAttention()
        self.lstm_attention = Attention(encoder_size*2)
        self.gru_attention = Attention(encoder_size*2)

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(2*encoder_size),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=2*encoder_size, out_features=16),
            nn.Sigmoid()
        )
        self.mlp2 = nn.Sequential(
            nn.BatchNorm1d(16+dim_num_feat),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=16+dim_num_feat, out_features=1),
            nn.Sigmoid()
        )
        self.apply(self._init_qa_weights)

    def forward(self, X, mask_X, length, num_feats):
        batch_size = X.size()[0]
        X = self.word_embedding(X)
        # For char embedding
        max_len = X.size()[1]
        #X = torch.squeeze(self.dropout1d(torch.unsqueeze(X, -1)), -1)
        X1 = self.LSTM(X, length)
        # X_context = self.self_attention(X1, extended_attention_mask)
        # attention_output = self.output(X_context, X1)
        # X1 = attention_output
        v = self.lstm_attention(X1, mask=mask_X)
        v = self.mlp(v)
        v = torch.cat([v, num_feats], dim=-1)
        v = self.mlp2(v)

        return v

    @staticmethod
    def _init_qa_weights(module):
        '''
        Initialize the weights of the qa model
        :param module:
        :return:
        '''
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.LSTM):
            nn.init.xavier_normal_(module.weight_ih_l0.data)
            nn.init.orthogonal_(module.weight_hh_l0.data)
            nn.init.constant_(module.bias_ih_l0.data, 0)
            nn.init.constant_(module.bias_hh_l0.data, 0)
            hidden_size = module.bias_hh_l0.data.shape[0]//4
            module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

            if module.bidirectional:
                nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
                nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
                nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
                nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
                module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0

    @staticmethod
    def _reshape_tensor(input_tensor):
        temp = []
        for sen in input_tensor:
            for sub_word in sen:
                temp.append(sub_word)
        return temp

    @staticmethod
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




class SpoSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob=0.1):
        super(SpoSelfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SpoSelfOutput(nn.Module):
    def __init__(self, hidden_size,hidden_dropout_prob):
        super(SpoSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# class BertAttention(nn.Module):
#     def __init__(self, config):
#         super(BertAttention, self).__init__()
#         self.self = BertSelfAttention(config)
#         self.output = BertSelfOutput(config)
#
#     def forward(self, input_tensor, attention_mask):
#         self_output = self.self(input_tensor, attention_mask)
#         attention_output = self.output(self_output, input_tensor)
#         return attention_output


class SPO_Model_Simple(nn.Module):
    def __init__(self,
                 vocab_size=20000,
                 word_embed_size=300,
                 init_embedding=None,
                 bidirectional=True,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.5,
                 seq_dropout=0.1,
                 pos_embed_size=100,
                 pos_dim=10):
        super(SPO_Model_Simple, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size,
                                           word_embed_size,
                                           padding_idx=0)
        self.pos_embedding = nn.Embedding(pos_embed_size, pos_dim, padding_idx=0)
        self.seq_dropout = seq_dropout
        self.embed_size = word_embed_size
        self.embed_size += pos_dim
        self.num_heads = 1
        if init_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.LSTM = LSTMEncoder(embed_size=1*(self.embed_size),
                                   encoder_size=encoder_size,
                                   bidirectional=bidirectional)
        self.GRU = GRUEncoder(embed_size=self.embed_size,
                                encoder_size=encoder_size,
                                bidirectional=bidirectional)
        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        self.attention = SoftAttention()
        self.lstm_attention = Attention(encoder_size*2)
        self.gru_attention = Attention(encoder_size*2)
        self.self_attention = SpoSelfAttention(hidden_size=256, num_attention_heads=self.num_heads)
        self.output = SpoSelfOutput(hidden_size=256, hidden_dropout_prob=0.1)

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(2*encoder_size),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=2*encoder_size, out_features=64),
            nn.Sigmoid()
        )
        self.mlp2 = nn.Sequential(
            nn.BatchNorm1d(64+dim_num_feat),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64+dim_num_feat, out_features=50),
            nn.Sigmoid()
        )

        # self.char_model =CharModel(embed_size=char_embed_size,
        #                            vocab_size=char_vocab_size,
        #                                encoder_size=encoder_size,
        #                                init_embedding=char_init_embedding,
        #                                bidirectional=bidirectional)
        self.apply(self._init_qa_weights)

    def forward(self, X, pos_tags, mask_X, length, num_feats):
        batch_size = X.size()[0]
        X = self.word_embedding(X)
        pos_X = self.pos_embedding(pos_tags)
        X = torch.cat([X, pos_X], dim=-1)
        attention_mask = mask_X
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        #X = torch.cat([X, attention_output], dim=-1)
        #X = torch.squeeze(self.dropout1d(torch.unsqueeze(X, -1)), -1)
        X1 = self.LSTM(X, length)
        X_context = self.self_attention(X1, extended_attention_mask)
        attention_output = self.output(X_context, X1)
        X1 = attention_output

        v = self.lstm_attention(X1, mask=mask_X)

        #X2 = self.GRU(X, length)
        #v2 = self.gru_attention(X2, mask=mask_X)

        # mask_X = mask_X.view(batch_size, X.size()[1], 1)
        # # avg should be masked
        # va_avg = (torch.sum(X1, dim=1)/torch.sum(mask_X)).view(batch_size, -1)
        # va_max = F.adaptive_max_pool1d(X1.transpose(1, 2), output_size=1).view(batch_size, -1)
        # # avg should be masked
        # #va_avg2 = (torch.sum(X2, dim=1) / torch.sum(mask_X)).view(batch_size, -1)
        # #va_max2 = F.adaptive_max_pool1d(X2.transpose(1, 2), output_size=1).view(batch_size, -1)

        #v = torch.cat([v, v2], dim=-1)

        v = self.mlp(v)
        v = torch.cat([v, num_feats], dim=-1)
        v = self.mlp2(v)

        return v

    @staticmethod
    def _init_qa_weights(module):
        '''
        Initialize the weights of the qa model
        :param module:
        :return:
        '''
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.LSTM):
            nn.init.xavier_normal_(module.weight_ih_l0.data)
            nn.init.orthogonal_(module.weight_hh_l0.data)
            nn.init.constant_(module.bias_ih_l0.data, 0)
            nn.init.constant_(module.bias_hh_l0.data, 0)
            hidden_size = module.bias_hh_l0.data.shape[0]//4
            module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0


class SPO_Model_Bert(nn.Module):
    def __init__(self,
                 vocab_size=20000,
                 word_embed_size=300,
                 init_embedding=None,
                 bidirectional=True,
                 encoder_size=64,
                 dim_num_feat=0,
                 dropout=0.5,
                 seq_dropout=0.1,
                 pos_embed_size=100,
                 pos_dim=10):
        super(SPO_Model_Bert, self).__init__()
        # self.word_embedding = nn.Embedding(vocab_size,
        #                                    word_embed_size,
        #                                    padding_idx=0)
        # self.pos_embedding = nn.Embedding(pos_embed_size, pos_dim, padding_idx=0)
        self.seq_dropout = seq_dropout
        # self.embed_size = word_embed_size
        # self.embed_size += pos_dim
        # if init_embedding is not None:
        # #     self.word_embedding.weight.data.copy_(torch.from_numpy(init_embedding))
        self.LSTM = LSTMEncoder(embed_size=768,
                                   encoder_size=encoder_size,
                                   bidirectional=bidirectional)
        # self.GRU = GRUEncoder(embed_size=self.embed_size,
        #                         encoder_size=encoder_size,
        #                         bidirectional=bidirectional)
        self.dropout1d = nn.Dropout2d(self.seq_dropout)
        self.attention = SoftAttention()
        self.lstm_attention = Attention(2*encoder_size)
        self.gru_attention = Attention(encoder_size*2)

        self.mlp = nn.Sequential(
            nn.BatchNorm1d(2*encoder_size),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=2*encoder_size, out_features=64),
            nn.Sigmoid()
        )
        self.mlp2 = nn.Sequential(
            nn.BatchNorm1d(64+dim_num_feat),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64+dim_num_feat, out_features=50),
            nn.Sigmoid()
        )
        bert_model = 'bert-base-chinese'
        self.bert = BertModel.from_pretrained(bert_model)
        self.use_layer = -1
        # self.char_model =CharModel(embed_size=char_embed_size,
        #                            vocab_size=char_vocab_size,
        #                                encoder_size=encoder_size,
        #                                init_embedding=char_init_embedding,
        #                                bidirectional=bidirectional)
        self.apply(self._init_qa_weights)

    def forward(self, token_tensor, pos_tags, mask_X, length, num_feats):
        batch_size = token_tensor.size()[0]
        self.bert.eval()
        with torch.no_grad():
            bert_outputs, _ = self.bert(token_tensor, attention_mask=(token_tensor > 0).long(), token_type_ids=None,
                                        output_all_encoded_layers=True)
        bert_outputs = torch.cat(bert_outputs[self.use_layer:], dim=-1)
        X1 = bert_outputs
        # X = self.word_embedding(X)
        # pos_X = self.pos_embedding(pos_tags)
        # X = torch.cat([X, pos_X], dim=-1)
        # #X = torch.squeeze(self.dropout1d(torch.unsqueeze(X, -1)), -1)
        X1 = self.LSTM(X1, length)

        v = self.lstm_attention(X1, mask=mask_X)
        v = self.mlp(v)
        v = torch.cat([v, num_feats], dim=-1)
        v = self.mlp2(v)

        return v

    @staticmethod
    def _init_qa_weights(module):
        '''
        Initialize the weights of the qa model
        :param module:
        :return:
        '''
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.LSTM):
            nn.init.xavier_normal_(module.weight_ih_l0.data)
            nn.init.orthogonal_(module.weight_hh_l0.data)
            nn.init.constant_(module.bias_ih_l0.data, 0)
            nn.init.constant_(module.bias_hh_l0.data, 0)
            hidden_size = module.bias_hh_l0.data.shape[0]//4
            module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0


# Mask attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        #print(hidden.shape, encoder_output.shape)
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs, mask):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs, mask)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs, mask)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs, mask)
        #print(attn_energies.shape)
        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        #print(F.softmax(attn_energies, dim=1).shape)
        #print(F.softmax(attn_energies, dim=1).unsqueeze(1).shape)
        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
