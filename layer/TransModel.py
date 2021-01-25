import torch
import torch.nn as nn

'''
Example of Transformer Modal
'''


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers,
                 n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()
        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList([EncoderLayer(hidden_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch_size, src_len]
        # src_mask = [batch_src, src_len]
        batch_size, src_len = src.shape[0], src.shape[1]

        # pos = [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # src = [batch_size, src_len, hidden_dim]
        src = self.dropout((self.tok_embedding(src) * self.scale) +
                           self.pos_embedding(pos))

        # src = [bacth_size, src_len, hidden_dim]
        for layer in self.layers:
            src = layer(src, src_mask)

        return src


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hidden_dim,
            n_heads,
            dropout,
            device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hidden_dim,
            pf_dim,
            dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch_size, src_len, hidden_dim]
        # src_mask = [batch_size, src_len]
        # _src == src = [batch_size, src_len, hidden_dim]
        # MultiHeadAttention
        _src, _ = self.self_attention(src, src, src, src_mask)
        # Position-wise Feedforward Layer¶
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)
        # dropout, residual and layer norm
        # src = [batch_size, src_len, hidden_dim]
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout, device):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.device = device
        self.dropout = nn.Dropout(dropout)

        assert hidden_dim % n_heads == 0
        self.head_dim = hidden_dim // n_heads

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        '''
        query = [batch_size, q_src_len, hidden_dim]
        key = [batch_size, k_src_len, hidden_dim]
        value = [batch_size, v_src_len, hidden_dim]
        '''
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        '''
        Q = [batch_size, n_heads, q_src_len, head_dim]
        K = [batch_size, n_heads, k_src_len, head_dim]
        V = [batch_size, n_heads, v_src_len, head_dim]
        '''
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # energy = [batch_size, n_heads, q_src_len, k_src_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        # attention = [batch_size, n_heads, q_src_len(v_src_len), k_src_len]
        attention = torch.softmax(energy, dim=-1)

        # x = [batch_size, n_heads, q_src_len(v_src_len), head_dim]
        x = torch.matmul(self.dropout(attention), V)

        # x = [batch_size, n_heads, head_dim, q_src_len(v_src_len)]
        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch_size, src_len, hidden_dim]
        x = x.view(batch_size, -1, self.hidden_dim)

        # x = [batch_size, src_len, hidden_dim]
        x = self.fc_o(x)

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch_size, src_len, pf_dim]
        x = self.fc_1(x)

        # x = [batch_size, src_len, hidden_dim]
        x = self.fc_2(x)

        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers,
                 n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()

        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList([DecoderLayer(hidden_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch_size, trg_len]
        # enc_src = [batch_size, src_len, hidden_dim]
        # trg_mask = [batch_size, trg_len]
        # src_mask = [batch_size, src_len]
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # pos = [batch size,_trg_len]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # trg = [batch_size, trg_len, hidden_dim]
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch_size, trg_len, hidden_dim]
        # attention = [batch_size, n_heads, trg_len, src_len]
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # output = [batch_size, trg_len, output_dim]
        # output = self.fc_out(trg)

        return trg, attention


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim,
                 dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hidden_dim,
            pf_dim,
            dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch_size, trg_len, hidden_dim]
        # enc_src = [batch_size, src_len, hidden_dim]
        # trg_mask = [batch_size, trg_len]
        # src_mask = [batch_size, src_len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        # trg = [batch_size, trg_len, hidden_dim]
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # dropout, residual connection and layer norm
        # trg = [batch size, trg len, hid dim]
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # dropout, residual and layer norm
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention


class SelectAttention(nn.Module):
    def __init__(self, emb_dim, length, opt):
        super().__init__()
        self.emb_dim = emb_dim
        self.length = length
        self.fc = nn.Linear(self.emb_dim, self.emb_dim)
        self.dropout = nn.Dropout(opt.dropout)

    def align(self, g_r1, g_r2, g_r1_mask, g_r2_mask):
        # g_r1 = [batch_size, 1, r1_len, emb_dim]
        # g_r2 = [batch_size, 1, r2_len, emb_dim]
        g_r1 = g_r1.unsqueeze(1)
        g_r2 = g_r2.unsqueeze(1)

        energy = torch.matmul(g_r1, g_r2.permute(0, 1, 3, 2))
        energy = energy.masked_fill(g_r2_mask == False, -1e10)
        attention = torch.softmax(energy, dim=-1)

        weights = torch.sum(attention, dim=-2).squeeze()
        _, sort_idx = torch.sort(weights, dim=-1, descending=True)
        sort_idx = sort_idx[:, :self.length]

        g_r2 = g_r2.squeeze()
        g_r2_mask = g_r2_mask.squeeze()
        x_mask = torch.gather(g_r2_mask, 1, sort_idx).unsqueeze(1).unsqueeze(1)

        sort_idx = sort_idx.unsqueeze(dim=-1).expand(sort_idx.shape[0], sort_idx.shape[1], g_r2.shape[-1])
        x = torch.gather(g_r2, 1, sort_idx)

        # x = torch.gather(g_r2, dim=2, sort_idx)

        # x, x_mask = [], []
        # for i in range(sort_idx.shape[0]):
        #     x.append(g_r2[i,:,sort_idx[i],:])
        #     x_mask.append(g_r2_mask[i,:,:,sort_idx[i]])
        # x = torch.stack(x).squeeze()
        # x_mask = torch.stack(x_mask)

        return x, x_mask

    def forward(self, g_r1, g_r2, g_r1_mask, g_r2_mask):
        x, x_mask = self.align(g_r1, g_r2, g_r1_mask, g_r2_mask)
        # x2 = self.align(g_r2, g_r1, g_r2_mask, g_r1_mask)

        # x = (x1[:,:self.length,:] + x2[:,:self.length,:])/2
        # x_mask = g_r1_mask[:,:,:,:self.length] & g_r2_mask[:,:,:,:self.length]
        # x = torch.cat([x1[:,:self.length,:], x2[:,:self.length,:]], dim=-2)
        # x_mask = torch.cat([g_r1_mask[:,:,:,:self.length], g_r2_mask[:,:,:,:self.length]], dim=-1)
        # x = torch.cat([x1, x2], dim=-2)
        # x_mask = torch.cat([g_r1_mask, g_r2_mask], dim=-1)

        return x, x_mask


class Transformer(nn.Module):
    def __init__(self, src_encoder, tgt_encoder, decoder, src_pad_idx, trg_pad_idx, opt):
        super().__init__()
        self.src_encoder = src_encoder
        self.tgt_encoder = tgt_encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = opt.device
        self.select_attn = SelectAttention(opt.word_vec_size, 10, opt)

    def make_src_mask(self, src):
        # src = [batch_size, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # sec_mask = [batch_size, 1, 1, src_len]
        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch_size, trg_len]
        # trg_pad_mask = [batch_size, 1, 1, trg_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        trg_len = trg.shape[1]
        # trg_sub_mask = [trg_len, trg_len]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_mask = [batch_size, 1, trg_len, trg_len]
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, keys, guide1, guide2, tgt):
        keys_mask = self.make_src_mask(keys)
        keys_emb = self.src_encoder(keys, keys_mask)

        guide1_mask = self.make_src_mask(guide1)
        guide1_emb = self.tgt_encoder(guide1, guide1_mask)

        guide2_mask = self.make_src_mask(guide2)
        guide2_emb = self.tgt_encoder(guide2, guide2_mask)

        add_emb, add_mask = self.select_attn(guide1_emb, guide2_emb, guide1_mask, guide2_mask)

        tgt = tgt[:, :-1]
        tgt_mask = self.make_trg_mask(tgt)

        enc_emb = torch.cat([keys_emb, add_emb], dim=1)
        enc_mask = torch.cat([keys_mask, add_mask], dim=-1)
        # enc_emb = keys_emb
        # enc_mask = keys_mask

        output, attention = self.decoder(tgt, enc_emb, tgt_mask, enc_mask)

        return output

nn.Module