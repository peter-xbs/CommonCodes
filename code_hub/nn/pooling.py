# _*_ coding:utf-8 _*_

"""
@Time: 2022/11/10 9:45 下午
@Author: jingcao
@Email: xinbao.sun@hotmail.com
使用多种Pooling的目的是增加BERT模型的多样性，考虑在模型集成中使用。
"""
import torch
from torch import nn


class SeqAvgPooling(nn.Module):
    """
    [batch, seq_len, hidden] => [batch, hidden] 序列平均池化
    """
    def __init__(self):
        super(SeqAvgPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask=None):
        if attention_mask is not None:
            # attention_mask [batch, seq_len]
            # input_mask_exp [batch, seq_len, 1]
            input_mask_exp = attention_mask.unsqueeze(-1)
            # input_mask_exp [batch, seq_len, hidden]
            # input_mask_exp = input_mask_exp.expand(last_hidden_state.size()).float() 由于广播机制，本步骤非必须
            # sum_embeddings: [batch, seq_len, hidden] => [batch, hidden]
            sum_embeddings = torch.sum(last_hidden_state * input_mask_exp, dim=1)
            # 等价于last_hidden_state.masked_fill(attention_mask==0, 0.0)
            sum_mask = input_mask_exp.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
        else:
            mean_embeddings = torch.mean(last_hidden_state, dim=1)
        return mean_embeddings


class SeqMaxPooling(nn.Module):
    # [batch, seq_len, hidden] => [batch, hidden] 序列最大池化
    def __init__(self):
        super(SeqMaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        """
        # [batch, seq_len, hidden], [batch, seq_len]
        :param last_hidden_state:
        :param attention_mask:
        :return:
        """
        if attention_mask is not None:
            mask = torch.ones_like(attention_mask) - attention_mask
            mask = mask * (-1e4)
            # 等价于last_hidden_state.masked_fill(attention_mask==0, -1e4)
            # [batch, seq_len, 1]
            mask = torch.unsqueeze(mask, dim=-1)
            out = last_hidden_state + mask
        else:
            out = last_hidden_state
        max_emb, max_indx = torch.max(out, dim=1)
        return max_emb
class SeqMinPooling(nn.Module):
    # [batch, seq_len, hidden] => [batch, hidden] 序列最小池化
    def __init__(self):
        super(SeqMinPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        """
        # [batch, seq_len, hidden], [batch, seq_len]
        :param last_hidden_state:
        :param attention_mask:
        :return:
        """
        if attention_mask is not None:
            mask = torch.ones_like(attention_mask) - attention_mask
            mask = mask * (1e4)
            # [batch, seq_len, 1]
            mask = torch.unsqueeze(mask, dim=-1)
            out = last_hidden_state + mask
        else:
            out = last_hidden_state
        min_emb, min_indx = torch.min(out, dim=1)
        return min_emb


class SeqWeightedLayerPooling(nn.Module):
    """
    effect: [layer_num, batch, seq_len, hidden] => [batch, seq_len, hidden]
    针对BERT来说, hidden_states共13层，按顺序从Embedding层到12个隐藏层，其中hidden_states[-1] == last_hidden_state
    """
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        """
        torch.expand: tensor size 和expand size优先匹配相同配好位置后再行expand，本例中unsqueeze 3次非常重要
        expand另外只是view上的改变，不会影响内存，也非复制
        相比之下 另一个torch.repeat_interleave的功能在此 https://blog.csdn.net/flyingluohaipeng/article/details/125039411
        :param num_hidden_layers: 一般为encoder layers + 1, 如bert-base为13
        :param layer_start: 加权起始层
        :param layer_weights: 每一层权重
        """
        super(SeqWeightedLayerPooling, self).__init__()
        if layer_weights:
            assert num_hidden_layers-layer_start == len(layer_weights)
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else torch.nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, ft_all_layers):
        # [layer_num, batch, seq_len, hidden]
        all_layer_embedding = torch.stack(ft_all_layers)
        # [keep_layer_num, batch, seq_len, hidden]
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]
        # weight -> [keep_layer_num, batch, seq_len, hidden]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        # [batch, seq_len, hidden]
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        return weighted_average


class SeqDotAttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super(SeqDotAttentionPooling).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1)
        )

    def forward(self, last_hidden_state, attention_mask=None):
        # [batch_size, seq_len, hidden], [batch_size, seq_len]
        w = self.attention(last_hidden_state).float()
        if attention_mask is not None:
            # [batch_size, seq_len, 1]
            w[attention_mask == 0] = float('-inf')
        w = torch.softmax(w, 1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings

class SeqAddAttentionPooling(nn.Module):
    def __init__(self, emb_size, units=None, use_additive_bias=False, use_attention_bias=False):
        super(SeqAddAttentionPooling).__init__()
        self.units = units if units else emb_size
        self.W = nn.Linear(emb_size, units, bias=False)
        self.U = nn.Linear(emb_size, units, bias=False)
        self.v = nn.Linear(units, 1, bias=False)
        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        if use_additive_bias:
            self.add_bias = nn.Parameter(torch.Tensor(units))
        if use_attention_bias:
            self.att_bias = nn.Parameter(torch.Tensor(1))

    def forward(self, last_hidden_state, attention_mask=None):
        # [batch_size, seq_len, hidden], [batch_size, seq_len]
        # [b, s, h] * [h, u] => [b, s, u]
        Q = self.W(last_hidden_state)
        K = self.U(last_hidden_state)
        if self.use_additive_bias:
            H = torch.tanh(Q + K + self.add_bias)
        else:
            H = torch.tanh(Q + K)
        # [b, s, u] * [u,1] = [b, s, 1]
        E = self.v(H)
        if self.use_attention_bias:
            E = E + self.att_bias
        if attention_mask is not None:
            E[attention_mask == 0] = float('-inf')
        alpha = torch.softmax(E, dim=1)
        attention_embeddings = torch.sum(alpha * last_hidden_state, dim=1)
        return attention_embeddings

class SeqValueAttentionPooling(nn.Module):
    """
    sanyuan's contribution
    """

    def __init__(self, embed_size, units=None, use_additive_bias=False, use_attention_bias=False):
        super(SeqValueAttentionPooling, self).__init__()
        self.units = units if units else embed_size
        self.embed_size = embed_size
        self.U = nn.Linear(self.embed_size, self.units, bias=False)
        self.V = nn.Linear(self.embed_size, self.units, bias=False)
        self.W = nn.Linear(self.units, self.embed_size, bias=False)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias

        if self.use_additive_bias:
            self.add_bias = nn.Parameter(torch.Tensor(self.units))
            nn.init.trunc_normal_(self.add_bias, std=0.01)
        if self.use_attention_bias:
            self.att_bias = nn.Parameter(torch.Tensor(self.embed_size))
            nn.init.trunc_normal_(self.att_bias, std=0.01)

    def forward(self, x, mask=None):
        # (B, Len, Embed) x (Embed, Units) = (B, Len, Units)
        Q = self.U(x)
        K = self.V(x)
        if self.use_additive_bias:
            H = torch.tanh(Q+K+self.add_bias)
        else:
            H = torch.tanh(Q+K)

        # (B, Len, Units) *(Units, Embed) => (B, Len, Embed)
        E = self.W(H)
        if self.use_attention_bias:
            E = E + self.att_bias

        # [B, L, E]
        if mask is not None:
            attention_probs = nn.Softmax(dim=1)(E + torch.unsqueeze((1.0 - mask) * -10000, dim=-1))
        else:
            attention_probs = nn.Softmax(dim=1)(E)
        # [B, E]
        x = torch.sum(attention_probs * x, dim=1)
        return x