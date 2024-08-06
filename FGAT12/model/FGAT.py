# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
# from transformers.modeling_bert import BertModel
from transformers import BertConfig
from .crf import CRF
from transformers import BertModel
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.autograd as autograd


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        # print("1111111",in_features,out_features)
        self.W = nn.Linear(in_features, out_features, bias=False)
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        # self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.a1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # print("223111111111",input.shape)
        h = self.W(input)
        # [batch_size, N, out_features]
        # print("sssssssssssss",h.shape)
        batch_size, N, _ = h.size()
        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, N)
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, N).transpose(1, 2)
        e = self.leakyrelu(middle_result1 + middle_result2)
        attention = e.masked_fill(adj == 0, -1e9)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        # h_prime = torch.cat((h_prime[:, :int(h_prime.size()[1]/2), :], h_prime[:, int(h_prime.size()[1]/2):, :]), dim=2)
        # h_prime = self.wei(h_prime)
        if self.concat:
            # print("wwwwwwwwwwwwwwww", h_prime.shape)
            # print("kkkkkkkkkkkkkkkk",F.elu(h_prime).shape)
            return F.elu(h_prime)
        else:
            # print("gggggggggggggggg",h_prime.shape)
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, layer):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layer = layer
        self.PositionwiseFeedForward = PositionwiseFeedForward(nhid * nheads, nclass * nheads)
        self.weight1 = nn.Parameter(torch.ones(nclass))
        self.weight2 = nn.Parameter(torch.ones(nclass))
        self.weight1.data = self.weight1.data.cuda()
        self.weight2.data = self.weight2.data.cuda()
        self.wei = nn.Linear(nclass * 2, nclass, bias=False)
        nn.init.xavier_uniform_(self.wei.weight, gain=1.414)
        if self.layer == 1:
            self.attentions = [GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha, concat=True) for _ in
                               range(nheads)]
        else:
            self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                               range(nheads)]
            self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.layer == 1:
            x = torch.stack([att(x, adj) for att in self.attentions], dim=2)
            x = x.sum(2)
            x = F.dropout(x, self.dropout, training=self.training)
            # print("qqqqqqqqqqqqqq",x.shape)
            return F.log_softmax(x, dim=2)
        else:
            x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.PositionwiseFeedForward(x)
            # print("aaaaaaaaaaaaaaa",x.shape)
            x = F.elu(self.out_att(x, adj))
            # print("pppppppppppppp",x.shape)
            x = torch.cat((x[:, :int(x.size()[1]/2), :], x[:, int(x.size()[1]/2):, :]), dim=2)
            x = self.wei(x)
            # x = self.weight1 * x[:, :int(x.size()[1]/2), :] + self.weight2 * x[:, int(x.size()[1]/2):, :]
            # x = torch.mul(x[:, :int(x.size()[1]/2), :],self.weight1) + torch.mul(x[:, int(x.size()[1]/2):, :],self.weight2)
            return F.log_softmax(x, dim=2)

def processgraph1(batch_size,length):
    g = torch.zeros(length, length)
    for j in range(length):
        for k in range(length):
            if (j >= 0 and j < int(length / 2)) and (k >= int(length / 2) and k < length) and j + int(
                    length / 2) == k:
                g[j][k] = 1
            elif (j >= int(length / 2) and j < length) and (k >= 0 and k < int(length / 2)) and j - int(
                    length / 2) == k:
                g[j][k] = 1
            elif j == k:
                g[j][k] = 1
    g = g.unsqueeze(0)
    gm = list()
    for i in range(batch_size):
        gm.append(g)
    graph = torch.cat(gm, dim=0)
    return graph

class FGAT(nn.Module):
    def __init__(self, data):
        super(FGAT, self).__init__()

        self.gpu = data.HP_gpu
        self.use_biword = data.use_bigram
        self.hidden_dim = data.HP_hidden_dim
        self.gaz_alphabet = data.gaz_alphabet
        self.gaz_emb_dim = data.gaz_emb_dim
        self.word_emb_dim = data.word_emb_dim
        self.biword_emb_dim = data.biword_emb_dim
        self.use_char = data.HP_use_char
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.use_count = data.HP_use_count
        self.num_layer = data.HP_num_layer
        self.use_bert = data.use_bert
        self.in_fc = nn.Linear(200*4+50, 768)
        self.n_feat = 768
        self.drop = 0.25
        self.alpha = 0.1
        self.gat_nhead = 3
        self.gat_layer = 2
        scale = np.sqrt(3.0 / self.gaz_emb_dim)
        crf_input_dim = data.label_alphabet.size() + 2
        # print("ffffffffff",data.label_alphabet.instance2index)
        # print("gggggggggg",data.label_alphabet.size())
        data.pretrain_gaz_embedding[0, :] = np.random.uniform(-scale, scale, [1, self.gaz_emb_dim])
        self.gat = GAT(self.hidden_dim*2, self.hidden_dim, crf_input_dim, self.drop, self.alpha, self.gat_nhead, self.gat_layer)
        self.lstm1 = nn.LSTM(self.n_feat, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.n_feat, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        if self.use_char:
            scale = np.sqrt(3.0 / self.word_emb_dim)
            data.pretrain_word_embedding[0, :] = np.random.uniform(-scale, scale, [1, self.word_emb_dim])

        self.gaz_embedding = nn.Embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.word_emb_dim)
        if self.use_biword:
            self.biword_embedding = nn.Embedding(data.biword_alphabet.size(), self.biword_emb_dim)

        if data.pretrain_gaz_embedding is not None:
            self.gaz_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_gaz_embedding))
        else:
            self.gaz_embedding.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.gaz_alphabet.size(), self.gaz_emb_dim)))

        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.word_emb_dim)))
        if self.use_biword:
            if data.pretrain_biword_embedding is not None:
                self.biword_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_biword_embedding))
            else:
                self.biword_embedding.weight.data.copy_(
                    torch.from_numpy(self.random_embedding(data.biword_alphabet.size(), self.word_emb_dim)))

        char_feature_dim = self.word_emb_dim + 4 * self.gaz_emb_dim
        if self.use_biword:
            char_feature_dim += self.biword_emb_dim

        if self.use_bert:
            char_feature_dim = char_feature_dim + 768

        print('total char_feature_dim {}'.format(char_feature_dim))

        self.drop = nn.Dropout(p=0.5)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.crf = CRF(data.label_alphabet_size, self.gpu)

        if self.use_bert:
            config = BertConfig.from_json_file('../CNNNERmodel/data/bert-base-chinese/config.json')
            self.bert_encoder = BertModel.from_pretrained('../CNNNERmodel/data/bert-base-chinese', config=config)
            for p in self.bert_encoder.parameters():
                p.requires_grad = False

        if self.gpu:
            self.gaz_embedding = self.gaz_embedding.cuda()
            self.word_embedding = self.word_embedding.cuda()
            self.gat = self.gat.cuda()
            self.lstm1 = self.lstm1.cuda()
            self.lstm2 = self.lstm2.cuda()
            if self.use_biword:
                self.biword_embedding = self.biword_embedding.cuda()
            self.crf = self.crf.cuda()
            self.in_fc = self.in_fc.cuda()
            self.drop =  self.drop.cuda()
            self.drop1 = self.drop1.cuda()
            self.drop2 = self.drop2.cuda()

            if self.use_bert:
                self.bert_encoder = self.bert_encoder.cuda()

    def get_tags(self, gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count, gaz_chars, gaz_mask_input,
                 gazchar_mask_input, mask, word_seq_lengths, batch_bert, bert_mask):

        batch_size = word_inputs.size()[0]
        seq_len = word_inputs.size()[1]
        max_gaz_num = layer_gaz.size(-1)
        gaz_match = []
        mask1 = word_inputs.ne(0)
        # print("ffffffffffff", word_inputs.shape)
        word_embs = self.word_embedding(word_inputs)
        word_embs = self.drop(word_embs)

        if self.use_char:
            # print("kkkkkkkkkkkkkkk",gaz_chars)
            gazchar_embeds = self.word_embedding(gaz_chars)

            gazchar_mask = gazchar_mask_input.unsqueeze(-1).repeat(1, 1, 1, 1, 1, self.word_emb_dim)
            gazchar_embeds = gazchar_embeds.data.masked_fill_(gazchar_mask.data, 0)  # (b,l,4,gl,cl,ce)

            # gazchar_mask_input:(b,l,4,gl,cl)
            gaz_charnum = (gazchar_mask_input == 0).sum(dim=-1, keepdim=True).float()  # (b,l,4,gl,1)
            gaz_charnum = gaz_charnum + (gaz_charnum == 0).float()
            gaz_embeds = gazchar_embeds.sum(-2) / gaz_charnum  # (b,l,4,gl,ce)

            gaz_embeds = self.drop(gaz_embeds)

        else:  # use gaz embedding
            # print("kkkkkkkkkkkkkkk",layer_gaz.shape)
            gaz_embeds = self.gaz_embedding(layer_gaz)
            # print("111111111111111",gaz_embeds.shape)

            gaz_embeds_d = self.drop(gaz_embeds)

            gaz_mask = gaz_mask_input.unsqueeze(-1).repeat(1, 1, 1, 1, self.gaz_emb_dim)

            gaz_embeds = gaz_embeds_d.data.masked_fill_(gaz_mask.data, 0)  # (b,l,4,g,ge)  ge:gaz_embed_dim

        if self.use_count:
            count_sum = torch.sum(gaz_count, dim=3, keepdim=True)  # (b,l,4,gn)
            count_sum = torch.sum(count_sum, dim=2, keepdim=True)  # (b,l,1,1)

            weights = gaz_count.div(count_sum)  # (b,l,4,g)
            weights = weights * 4
            weights = weights.unsqueeze(-1)
            gaz_embeds = weights * gaz_embeds  # (b,l,4,g,e)
            gaz_embeds = torch.sum(gaz_embeds, dim=3)  # (b,l,4,e)

        else:
            gaz_num = (gaz_mask_input == 0).sum(dim=-1, keepdim=True).float()  # (b,l,4,1)
            gaz_embeds = gaz_embeds.sum(-2) / gaz_num  # (b,l,4,ge)/(b,l,4,1)

        gaz_embeds_cat = gaz_embeds.view(batch_size, seq_len, -1)  # (b,l,4*ge)

        word_input_cat = torch.cat([word_embs, gaz_embeds_cat], dim=-1)  # (b,l,we+4*ge)

        ### cat bert feature
        if self.use_bert:
            seg_id = torch.zeros(bert_mask.size()).long().cuda()
            outputs = self.bert_encoder(batch_bert, bert_mask, seg_id)
            outputs = outputs[0][:, 1:-1, :]

            gaz_embeds_cat = self.in_fc(word_input_cat) # outputs:bert gaz_embeds_cat:BMES

        feature_out_d1, _ = self.lstm1(outputs)
        feature_out_d2, _ = self.lstm2(gaz_embeds_cat)
        feature_out_d1 = self.drop1(feature_out_d1)
        feature_out_d2 = self.drop2(feature_out_d2)

        gat_input = torch.cat((feature_out_d1, feature_out_d2), dim=1)
        graph = processgraph1(gat_input.size()[0],gat_input.size()[1])
        tags = self.gat(gat_input, graph.cuda())

        return tags, gaz_match

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, layer_gaz, gaz_count,
                                gaz_chars, gaz_mask, gazchar_mask, mask, batch_label, batch_bert, bert_mask):

        tags, _ = self.get_tags(gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count, gaz_chars, gaz_mask,
                                gazchar_mask, mask, word_seq_lengths, batch_bert, bert_mask)

        total_loss = self.crf.neg_log_likelihood_loss(tags, mask, batch_label)
        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return total_loss, tag_seq

    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, layer_gaz, gaz_count, gaz_chars, gaz_mask,
                gazchar_mask, mask, batch_bert, bert_mask):

        tags, gaz_match = self.get_tags(gaz_list, word_inputs, biword_inputs, layer_gaz, gaz_count, gaz_chars, gaz_mask,
                                        gazchar_mask, mask, word_seq_lengths, batch_bert, bert_mask)

        scores, tag_seq = self.crf._viterbi_decode(tags, mask)

        return tag_seq, gaz_match






