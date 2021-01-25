import torch
import numpy as np
import torch.nn as nn
import layer.modules
from torch.autograd import Variable

from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class Encoder(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        
        assert opt.enc_rnn_size % self.num_directions == 0
        self.hidden_size = opt.enc_rnn_size // self.num_directions
        self.input_size = opt.word_vec_size
        
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(dicts.size(),
                                      self.input_size,
                                      padding_idx=layer.Constants.PAD)
        self.rnn = nn.GRU(self.input_size, self.hidden_size,
                          num_layers=opt.layers,
                          dropout=opt.dropout,
                          bidirectional=opt.brnn)
    def forward(self, x, hidden=None):
        # input = []
        lengths = x[-1].data.view(-1).tolist()
        
        embed = self.embedding(x[0])
        embed = pack(embed, lengths)
        
        hiddens, hidden_t = self.rnn(embed, hidden)
        if isinstance(x, tuple):
            hiddens = unpack(hiddens)[0]
        return hidden_t, hiddens


class EditEncoder(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.hidden_size = opt.enc_rnn_size
        
        super(EditEncoder, self).__init__()
        self.embedding = nn.Embedding(dicts.size(),
                                      opt.word_vec_size,
                                      padding_idx=layer.Constants.PAD)
    def forward(self, x):
        embed = self.embedding(x[0])
        hidden_t = torch.sum(embed, x)

        return hidden_t
        

class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1

class EditAttDecoder(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.enc_rnn_size + opt.word_vec_size * 2
        self.maxout_pool_size = opt.maxout_pool_size
        self.hidden_size = opt.dec_rnn_size
        
        super(EditAttDecoder, self).__init__()
        self.embedding = nn.Embedding(dicts.size(),
                                      opt.word_vec_size,
                                      padding_idx=layer.Constants.PAD)
        self.rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)
        self.attn1 = layer.modules.ConcatAttention(opt.enc_rnn_size,
                                                     opt.dec_rnn_size,
                                                     opt.att_vec_size)
        self.attn2 = layer.modules.ConcatAttention(opt.word_vec_size,
                                                     opt.enc_rnn_size,
                                                     opt.word_vec_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.readout = nn.Linear((opt.enc_rnn_size
                                  + opt.dec_rnn_size
                                  + opt.word_vec_size
                                  + opt.word_vec_size * 2 # edit vector length
                                  ), opt.dec_rnn_size)
        self.maxout = layer.modules.MaxOut(opt.maxout_pool_size)
        
    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, tgt, enc_hidden, src_ins, src_del, context, src_pad_mask, init_att, ins_pad_mask, del_pad_mask):
        '''tgt = [tgt_len, batch_size]'''
        emb = self.embedding(tgt)
        '''emb = [tgt_len, batch_size, embed_size]'''
        
        self.attn2.applyMask(ins_pad_mask)
        ins_embed = self.embedding(src_ins)
        '''
        context[-1](target) = [batch_size, encoder_hidden_size]
        ins_embed = [batch_size, embedding_size]
        ins_hidden = [batch_size, embedding_size]'''
        ins_hidden = self.attn2(context[-1], ins_embed.transpose(0, 1), None)[0]
        
        self.attn2.applyMask(del_pad_mask)
        del_emebd = self.embedding(src_del)
        '''
        context[-1](target) = [batch_size, encoder_hidden_size]
        del_embed = [batch_size, embedding_size]
        del_hidden = [batch_size, embedding_size]'''
        del_hidden = self.attn2(context[-1], del_emebd.transpose(0, 1), None)[0]
        #del_hidden = torch.sum(wordEmb, dim=0)

        g_outputs = []
        cur_context = init_att
#         conv1d = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=context.size(0))
#         cur_context = conv1d(context.permute(1,2,0)).squeeze(-1)
#         print(cur_context.shape, init_att.shape, context.shape)
        self.attn1.applyMask(src_pad_mask)
        precompute = None
        '''emb = [tgt, batch_size, embedding_size]'''
        # split 参数：1表示大小为1， dim=0表示在0维度上划分
        for emb_t in emb.split(1, dim=0):
            '''emb_t(==tgt_emb) = [1, batch_size, embedding_size]'''
            emb_t = emb_t.squeeze(0)
            input_emb = emb_t
            if self.input_feed:
                '''input_emb = [batch_size, mix_size(embedding_size
                                                    +encoder_hidden_size/
                                                    +embedding_size
                                                    +embedding_size)]'''
                input_emb = torch.cat([emb_t, cur_context, ins_hidden, del_hidden], dim=1)
            '''
            output = [batch_size, decoder_hidden_size]
            hidden = [1, batch_size, decoder_hidden_size]'''
            output, hidden = self.rnn(input_emb, enc_hidden)
            
            '''
            output(target) = [batch_size, decoder_hidden_size]
            context = [tgt_len, batch_size, encoder_hidden_size]
            cur_context = [batch_size, encoder_hidden_size]
            '''
            cur_context, attn, precompute = self.attn1(output, context.transpose(0, 1), precompute)
            '''
            输入 = [batch_size, mix_size(embedding_size
                                       +decoder_hidden_size
                                       +encoder_hidden_size
                                       +embedding_size
                                       +embedding_size)]
            输出 = readout = [batch_size, decoder_hidden_size]'''
            readout = self.readout(torch.cat((emb_t, output, cur_context, ins_hidden, del_hidden), dim=1))
            '''maxout = [batch_size, decoder_hidden_size/2]????????'''
            maxout = self.maxout(readout)
            output = self.dropout(maxout)
            g_outputs += [output]
        '''g_outputs = [tgt_len, batch_size, decoder_hidden_size/2]'''
        g_outputs = torch.stack(g_outputs)

        return g_outputs, hidden, attn, cur_context
        

class DecInit(nn.Module):
    def __init__(self, opt):
        super(DecInit, self).__init__()
        self.num_directions = 2 if opt.brnn else 1
        assert opt.enc_rnn_size % self.num_directions == 0
        self.enc_rnn_size = opt.enc_rnn_size
        self.dec_rnn_size = opt.dec_rnn_size
        self.initer = nn.Linear(self.enc_rnn_size // self.num_directions, self.dec_rnn_size)
        self.tanh = nn.Tanh()

    def forward(self, last_enc_h):

        return self.tanh(self.initer(last_enc_h))


class IDEditModel(nn.Module):
    def __init__(self, encoder, editEncoder, decoder, decIniter):
        super(IDEditModel, self).__init__()
        
        self.encoder = encoder
        self.editEncoder = editEncoder
        self.decoder = decoder
        self.decIniter = decIniter

    
    def make_init_att(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.encoder.hidden_size * self.encoder.num_directions)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)
        
    
    def forward(self, x):
        """
        src = [src_len, batch_size]
        src_ins = [src_ins_len, batch_size]
        src_del = [src_del_len, batch_size]"""
        src, src_ins, src_del = x[0], x[1][0], x[2][0]
        src_pad_mask = Variable(src[0].data.eq(layer.Constants.PAD).transpose(0, 1).float(),
                                requires_grad=False,
                                volatile=False)
        src_ins_len = Variable(src_ins.data.eq(layer.Constants.PAD).transpose(0, 1).float(),
                               requires_grad=False,
                               volatile=False)
        src_del_len = Variable(src_del.data.eq(layer.Constants.PAD).transpose(0, 1).float(),
                               requires_grad=False,
                               volatile=False)
        tgt = x[3][0][:-1]
        
        """
        enc_hidden = [2, batch_size, encoder_hidden_size/2]
        context = [src_len, batch_size, encoder_hidden_size]"""
        enc_hidden, context = self.encoder(src)

        '''init_atta = [batch_size, encoder_hidden_size], 但是这有什么用呢？'''
        init_att = self.make_init_att(context)
        
        '''enc_hidden = [1, batch_size, decoder_hidden_size(== encoder_hidden_size)], 但是这有什么用呢？'''
        enc_hidden = self.decIniter(enc_hidden[1]).unsqueeze(0)
        
        '''
        输入:
        tgt = [tgt_len ,batch_size]
        enc_hidden = [1, batch_size, decoder_hidden_size]
        src_ins = [src_ins_len, batch_size] | src_del = [src_del_len, batch_size]
        context = [src_len, batch_size, encder_hidden]
        src_pad_mask = [batch_size, src_len], 标志哪些位置为PAD
        init_att = [batch_size, encoder_hidden_size]
        src_ins_len = [batch_size, src_ins_len] | src_del_len = [batch_size, src_del_len]'''
        g_out, dec_hidden, _attn, _attention_vector = self.decoder(tgt, enc_hidden,
                                                                   src_ins, src_del,
                                                                   context, src_pad_mask,
                                                                   init_att, src_ins_len,
                                                                   src_del_len)
        return g_out
