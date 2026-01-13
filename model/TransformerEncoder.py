# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 19:44:36 2022

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)
    
          
class ScaleDotProductAttention(nn.Module):
    
    def __init__(self,scale,attn_dropout=0.1):
        super(ScaleDotProductAttention,self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attn_dropout)
        
    def forward(self,q,k,v,mask):
        attn = torch.matmul(q/self.scale, k.transpose(2,3))
        attn = attn + mask
        attn = self.dropout(F.softmax(attn , dim=-1))
        output = torch.matmul(attn, v)        
        return output,attn
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_head, dim, d_k, d_v, dropout = 0.1):
        super(MultiHeadAttention,self).__init__()
        
        self.n_head = n_head
        self.d_v = d_v
        self.d_k = d_k
        
        self.w_qs = nn.Linear(dim, n_head * d_k, bias = False)
        self.w_ks = nn.Linear(dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(dim, n_head * d_v, bias=False)
        
        self.fc = nn.Linear(n_head * d_v, dim, bias=False)

        self.attention = ScaleDotProductAttention(scale=d_k ** 0.5,attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        initialize_weight(self.w_qs)
        initialize_weight(self.w_ks)
        initialize_weight(self.w_vs)
        initialize_weight(self.fc)

    def forward(self, q, k, v, mask):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, attn = self.attention(q, k, v, mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        return q, attn
    
class MultiHeadAttention_fr(nn.Module):
    
    def __init__(self, n_head, dim, d_k, d_v, dropout = 0.1):
        super(MultiHeadAttention_fr,self).__init__()
        
        self.n_head = n_head
        self.d_v = d_v
        self.d_k = d_k
        
        self.w_qs = nn.Linear(dim, n_head * d_k, bias = False)
        self.w_ks = nn.Linear(dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(dim, n_head * d_v, bias=False)
        
        self.fc = nn.Linear(n_head * d_v, dim, bias=False)

        self.attention = ScaleDotProductAttention(scale=d_k ** 0.5,attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        initialize_weight(self.w_qs)
        initialize_weight(self.w_ks)
        initialize_weight(self.w_vs)
        initialize_weight(self.fc)

    def forward(self, q, k, v, mask):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # Transpose for attention dot product: b x n x frequency x channel 
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
        q, attn = self.attention(q, k, v, mask)
        
        q = q.transpose(2,3)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        return q, attn

class MultiHeadAttention_mx(nn.Module):
    
    def __init__(self, n_head, dim, d_k, d_v, dropout = 0.1):
        super(MultiHeadAttention_mx,self).__init__()
        
        self.n_head = n_head
        self.d_v = d_v
        self.d_k = d_k
        
        self.w_qs = nn.Linear(dim, n_head * d_k, bias = False)
        self.w_ks = nn.Linear(dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(dim, n_head * d_v, bias=False)
        
        self.fc = nn.Linear(n_head * d_v, dim, bias=False)

        self.attention = ScaleDotProductAttention(scale=d_k ** 0.5,attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        initialize_weight(self.w_qs)
        initialize_weight(self.w_ks)
        initialize_weight(self.w_vs)
        initialize_weight(self.fc)

    def forward(self, q, k, v, mask):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        # 按照奇偶划分不同的头的识别模式
        i = torch.arange(n_head)
        i_s = torch.where(i%2==0)[0]
        i_f = torch.where(i%2==1)[0]
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # spatial pattern
        q_s, k_s, v_s = q[:,i_s], k[:,i_s], v[:,i_s]
        # frequency pattern
        q_f, k_f, v_f = q[:,i_f], k[:,i_f], v[:,i_f]
        # Transpose for attention dot product: b x n x frequency x channel 
        q_f, k_f, v_f = q_f.transpose(2, 3), k_f.transpose(2, 3), v_f.transpose(2, 3)
        q_s, attn_s = self.attention(q_s, k_s, v_s, mask[0])
        q_f, attn_f = self.attention(q_f, k_f, v_f, mask[1])
        
        # Transpose for attention dot product: b x n x lq x dv
        q_f = q_f.transpose(2,3)
        
        # concate different pattern
        q = torch.cat((q_s,q_f),1)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        
        attn = (attn_s,attn_f)
        return q, attn

class MultiHeadAttention_ab1(nn.Module):
    
    def __init__(self, n_head, dim, d_k, d_v, dropout = 0.1):
        super(MultiHeadAttention_ab1,self).__init__()
        
        self.n_head = n_head
        self.d_v = d_v
        self.d_k = d_k
        
        self.w_qs = nn.Linear(dim, n_head * d_k, bias = False)
        self.w_ks = nn.Linear(dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(dim, n_head * d_v, bias=False)
        
        self.fc = nn.Linear(n_head * d_v, dim, bias=False)

        self.attention = ScaleDotProductAttention(scale=d_k ** 0.5,attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        initialize_weight(self.w_qs)
        initialize_weight(self.w_ks)
        initialize_weight(self.w_vs)
        initialize_weight(self.fc)

    def forward(self, q, k, v, mask):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        # 按照奇偶划分不同的头的识别模式
        i = torch.arange(n_head)
        i_s = torch.where(i%2==0)[0]
        i_f = torch.where(i%2==1)[0]
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # spatial pattern
        q_s, k_s, v_s = q[:,i_s], k[:,i_s], v[:,i_s]
        # frequency pattern
        q_f, k_f, v_f = q[:,i_f], k[:,i_f], v[:,i_f]
        # Transpose for attention dot product: b x n x frequency x channel 
        q_f, k_f, v_f = q_f.transpose(2, 3), k_f.transpose(2, 3), v_f.transpose(2, 3)
        q_s, attn_s = self.attention(q_s, k_s, v_s, mask[0])
        # q_f, attn_f = self.attention(q_f, k_f, v_f, mask[1])
        
        # Transpose for attention dot product: b x n x lq x dv
        q_f = q_f.transpose(2,3)
        
        # concate different pattern
        q = torch.cat((q_s,q_f),1)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        
        attn = attn_s
        return q, attn

class MultiHeadAttention_ab2(nn.Module):
    
    def __init__(self, n_head, dim, d_k, d_v, dropout = 0.1):
        super(MultiHeadAttention_ab2,self).__init__()
        
        self.n_head = n_head
        self.d_v = d_v
        self.d_k = d_k
        
        self.w_qs = nn.Linear(dim, n_head * d_k, bias = False)
        self.w_ks = nn.Linear(dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(dim, n_head * d_v, bias=False)
        
        self.fc = nn.Linear(n_head * d_v, dim, bias=False)

        self.attention = ScaleDotProductAttention(scale=d_k ** 0.5,attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        initialize_weight(self.w_qs)
        initialize_weight(self.w_ks)
        initialize_weight(self.w_vs)
        initialize_weight(self.fc)

    def forward(self, q, k, v, mask):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        # 按照奇偶划分不同的头的识别模式
        i = torch.arange(n_head)
        i_s = torch.where(i%2==0)[0]
        i_f = torch.where(i%2==1)[0]
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # spatial pattern
        q_s, k_s, v_s = q[:,i_s], k[:,i_s], v[:,i_s]
        # frequency pattern
        q_f, k_f, v_f = q[:,i_f], k[:,i_f], v[:,i_f]
        # Transpose for attention dot product: b x n x frequency x channel 
        q_f, k_f, v_f = q_f.transpose(2, 3), k_f.transpose(2, 3), v_f.transpose(2, 3)
        # q_s, attn_s = self.attention(q_s, k_s, v_s, mask[0])
        q_f, attn_f = self.attention(q_f, k_f, v_f, mask[1])
        
        # Transpose for attention dot product: b x n x lq x dv
        q_f = q_f.transpose(2,3)
        
        # concate different pattern
        q = torch.cat((q_s,q_f),1)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        
        attn = attn_f
        return q, attn
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.dropout = nn.Dropout(dropout)
        initialize_weight(self.w_1)
        initialize_weight(self.w_2)

    def forward(self, x):

        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self,dim, hidden_dim,n_head, d_k, d_v, dropout = 0., mode = 1):
        super(EncoderLayer,self).__init__()
        self.self_attention_norm = nn.LayerNorm(dim)
        self.ff = PositionwiseFeedForward(dim, hidden_dim, dropout = dropout)
        if mode == 1:
            self.slf_attn = MultiHeadAttention(n_head, dim, d_k, d_v, dropout=dropout)
        elif mode == 2:
            self.slf_attn = MultiHeadAttention_fr(n_head, dim, d_k, d_v, dropout=dropout)
        elif mode == 3:
            self.slf_attn = MultiHeadAttention_mx(n_head, dim, d_k, d_v, dropout=dropout)
        elif mode == 4:
            self.slf_attn = MultiHeadAttention_ab1(n_head, dim, d_k, d_v, dropout=dropout)
        else:
            self.slf_attn = MultiHeadAttention_ab2(n_head, dim, d_k, d_v, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(dim)
    def forward(self,x,mask,attn_show = False):
        
        y = self.self_attention_norm(x)
        y,attn = self.slf_attn(y,y,y,mask)
        x = x + y
        
        y = self.ffn_norm(x)
        y = self.ff(y)
        x = x + y
        
        if attn_show is True:
            return x,attn
        else:
            return x
        
class Encoder(nn.Module):
    def __init__(self,dim, hidden_dim,n_head, d_k, d_v, n_layers, dropout = 0., mode = 1):
        super(Encoder,self).__init__()
        encoders = [EncoderLayer(dim, hidden_dim,n_head,d_k, d_v, dropout, mode)
                    for _ in range(n_layers)]
        
        self.layers = nn.ModuleList(encoders)
        self.last_norm = nn.LayerNorm(dim)
        self.mode = mode
     
    def forward(self, inputs, mask,attn_show = False):
        encoder_output = inputs
        mode = self.mode
        if mode != 1 and mode != 2:
            mask_s = self.get_extended_attention_mask(mask[0])
            mask_f = self.get_extended_attention_mask(mask[1])
            mask_s = mask_s.to(encoder_output.device)
            mask_f = mask_f.to(encoder_output.device)
            mask = (mask_s,mask_f)
        else:
            mask = self.get_extended_attention_mask(mask)
            mask = mask.to(encoder_output.device)
        if attn_show:   
            attn_list = []
            for enc_layer in self.layers:
                encoder_output,attn = enc_layer(encoder_output, mask, attn_show)
                attn_list.append(attn)
            return self.last_norm(encoder_output),attn_list
        else:
            for enc_layer in self.layers:
                encoder_output = enc_layer(encoder_output, mask, attn_show)
                
            return self.last_norm(encoder_output)
    
    def get_extended_attention_mask(self,mask):
        extended_attention_mask = mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

if __name__ == "__main__":
    data = torch.randn(12,775,10)
    E = Encoder(10,10*4,3,10,10,2,0)
    mask = torch.ones(775)
    mask = mask.repeat(12,1)
    o = E(data,mask)
    print(o.shape)
    
    #test return attn
    o,a = E(data,mask,True)
    print(a)
    