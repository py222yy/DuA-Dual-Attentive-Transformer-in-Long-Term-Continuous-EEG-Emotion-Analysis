# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:26:34 2022

@author: user
"""
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from .TransformerEncoder import Encoder
from .adv_loss import AdversarialLoss
import math
from scipy.stats import truncnorm


class Feature_Extractors_WA(nn.Module):
    def __init__(self,input_dim,channel,n_head, d_k, d_v, n_layers, mode,dropout = 0.):
        super(Feature_Extractors_WA,self).__init__()
        self.spatial_transformer = Encoder(input_dim,input_dim*4,n_head, d_k, d_v, n_layers, dropout,mode)
        # self.frequency_transformer = Encoder(channel,channel*4,n_head, channel, channel, n_layers, dropout)

        #can try add position encoding
        trunc_array = truncnorm.rvs(-2, 2, loc=0, scale=0.02, size=[1,63,input_dim], random_state=None)
        self.inner_pos = nn.Parameter(torch.tensor(trunc_array).to(torch.float))
        self.mode = mode
        
    def forward(self,x,attn_show = False):
        mode = self.mode
        #require input shape 3d [bs*s,c,f]
        n_batch_size = x.shape[0]
        n_channel = x.shape[1]
        n_feature = x.shape[2]
        mask_s = torch.ones(n_channel).repeat(n_batch_size,1)
        mask_feature = torch.ones(n_feature).repeat(n_batch_size,1)
        if mode == 1:
            mask = mask_s
        elif mode == 2:
            mask = mask_feature
        else:
            mask = (mask_s,mask_feature)
            
        x = x + self.inner_pos
        if attn_show:
            y,a = self.spatial_transformer(x,mask,attn_show)
            return y,a
        else:
              y = self.spatial_transformer(x,mask)
              return y
        

class FeedForwardNet(nn.Module):
    def __init__(self,hidden_dim):
        super(FeedForwardNet,self).__init__()
        self.fc = nn.Linear(hidden_dim,hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.layer = torch.nn.Sequential(
                self.fc,
                self.norm,
                nn.GELU()
                )
          
    def forward(self,x):           
        y = self.layer(x)
        return y 
    
class Feature_Extractors_WOA(nn.Module):
    def __init__(self,input_dim,hidden_dim,n_layers, dropout = 0.):
        super(Feature_Extractors_WOA,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        layer = [FeedForwardNet(hidden_dim) for _ in range(n_layers)]
        self.fcnet = nn.ModuleList(layer)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        #require input shape 2d [bs*s,c*f]
        #if need can add res
        y = self.fc1(x)
        for fclayer in self.fcnet:
            y = fclayer(y)   
        y = self.dropout(y)
        return y

class second_base_model(nn.Module):
    def __init__(self,aflag,channel,n_class,mode,**kwargs):
        super(second_base_model,self).__init__()
        
        self.hidden_dim = kwargs['hidden_dim']
        self.aflag = aflag
        self.channel = channel
        if aflag:
            kwargs.pop('hidden_dim')
            kwargs.update({'channel':self.channel})
            self.feature_extractor = Feature_Extractors_WA(mode=mode,**kwargs)
            #未来要修改
            self.downfc = nn.Linear(kwargs['input_dim']*self.channel,self.hidden_dim)
            # self.downfc = nn.Linear(kwargs['input_dim']*channel,self.hidden_dim)
            #
            self.norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.feature_extractor = Feature_Extractors_WOA(**kwargs)
        self.classifier = nn.Linear(self.hidden_dim,n_class)
        
    def forward(self,x,attn_show = False):
        #require input 3d [b,c,f]
        dim = x.shape[-1]
        if self.aflag:
            if attn_show:
                x,a = self.feature_extractor(x,attn_show)
            else:
                x = self.feature_extractor(x)
            x = x.reshape(-1,self.channel*dim)
            x = self.downfc(x)
            x = F.gelu(self.norm(x))
        else:
            x = x.reshape(-1,self.channel*dim)
            x = self.feature_extractor(x)
        feature = x
        logics = self.classifier(x)
        logics = logics.squeeze(-1)
        if attn_show:
            return logics,feature,a
        else:
            return logics,feature
    
class trasfernet_second(nn.Module):
    def __init__(self,aflag,channel,n_class,mode,max_iter=1000,**kwargs):
        super(trasfernet_second,self).__init__()
        self.net = second_base_model(aflag,channel,n_class,mode,**kwargs)
        self.adapt_loss = AdversarialLoss(max_iter=max_iter,**kwargs)
        self.n_class = n_class
        self.aflag = aflag
        if n_class > 2:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.BCELoss()
            
    def forward(self,source,target,source_label):
        source,source_feature = self.net(source)
        target,target_feature = self.net(target)
        if self.n_class > 2:
            clf_loss = self.criterion(source,source_label)
        else:
            source = torch.sigmoid(source)
            clf_loss = self.criterion(source,source_label)
        
        transfer_loss = self.adapt_loss(source_feature, target_feature)
        return clf_loss,transfer_loss
    
    def predict(self,x,attn_show = False):
        if self.aflag:
            if attn_show:
                logics,f,a = self.net(x,attn_show)
                if self.n_class > 2:
                    return logics,a
                else:
                    return torch.sigmoid(logics),a
            else:
                logics,_ = self.net(x)
                if self.n_class > 2:
                    return logics
                else:
                    return torch.sigmoid(logics)
        else:
            logics,_ = self.net(x)
            if self.n_class > 2:
                return logics
            else:
                return torch.sigmoid(logics)
        
        
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.net.parameters(), 'lr': 1.0 * initial_lr},
        ]
        
        params.append(
            {'params': self.adapt_loss.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
        )
        
        return params
    
class trasfernet_trial(nn.Module):
    def __init__(self,aflag,channel,n_class,sn_head,sn_layers,pool_flag,max_iter=1000,Lstm_flag=False,mode=3,**kwargs):
        super(trasfernet_trial,self).__init__()
        self.hidden_dim = kwargs["hidden_dim"]
        self.dropout = kwargs['dropout']
        self.aflag = aflag
        self.net = second_base_model(aflag,channel,n_class,mode,**kwargs)
        
        self.n_class = n_class
        self.Lstm_flag = Lstm_flag
        if Lstm_flag:
            self.seq_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, dropout=self.dropout,batch_first=True,bidirectional=True)
            self.classifier = nn.Linear(self.hidden_dim*2,n_class)
            kwargs["hidden_dim"] = kwargs["hidden_dim"]*2
            self.adapt_loss = AdversarialLoss(max_iter=max_iter,**kwargs)
        else:
            self.seq_transformer = Encoder(self.hidden_dim,self.hidden_dim*4,sn_head, self.hidden_dim, self.hidden_dim, sn_layers, self.dropout,1)
            self.classifier = nn.Linear(self.hidden_dim,n_class)
            self.adapt_loss = AdversarialLoss(max_iter=max_iter,**kwargs)
            # can add pe and cls token
            num_timescales = self.hidden_dim // 2
            max_timescale = 10000.0
            min_timescale = 1.0
            log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
            inv_timescales = min_timescale * torch.exp(
                torch.arange(num_timescales, dtype=torch.float32) *
                -log_timescale_increment)
            self.inv_timescales = inv_timescales
            
            self.pool_flag = pool_flag
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
            self.cls_mask = torch.ones(1,1)
        if n_class > 2:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.BCELoss()
        
    def forward(self,source,target,source_label,source_mask,target_mask):
        # require input 4D [batch,seq_len,c,f]
        source_shape = source.shape
        target_shape = target.shape
        #down 3D [batch*seq_len,c,f]       
        source = source.reshape(-1,source_shape[2],source_shape[3])
        target = target.reshape(-1,target_shape[2],target_shape[3])
        # non_zero index
        source_index = torch.where(source_mask.reshape(-1)!=0)[0]
        target_index = torch.where(target_mask.reshape(-1)!=0)[0]
        # feature 2D [batch*seq,hidden_dim]
        _,source_feature = self.net(source[source_index])
        _,target_feature = self.net(target[target_index])
        
        # _,source_feature = self.net(source,mode)
        # _,target_feature = self.net(target,mode)
        
        # transfer_loss = self.adapt_loss(source_feature, target_feature)
        
        #up 3D [batch,seq_len,c,hidden_dim]
        source_feature_new = torch.zeros(source.shape[0],source_feature.shape[1]).to(source.device)
        target_feature_new = torch.zeros(target.shape[0],target_feature.shape[1]).to(target.device)
        source_feature_new[source_index] = source_feature
        target_feature_new[target_index] = target_feature
        source_feature = source_feature_new.reshape(source_shape[0],source_shape[1],-1)
        target_feature = target_feature_new.reshape(target_shape[0],target_shape[1],-1)
        # source_feature = source_feature.reshape(source_shape[0],source_shape[1],-1)
        # target_feature = target_feature.reshape(target_shape[0],target_shape[1],-1)
        
        # return source_feature,target_feature
        # input seq transformer out seq 3D [batch,seq_len,c,f] pool mean cls seq[0]
        if self.Lstm_flag:
            # source_output
            source_feature, (source_hn, cn) = self.seq_lstm(source_feature)
            # hn shape is [2,Batch,hidden_size]
            sz_b = source_feature.shape[0]
            source_hn = source_hn.permute(1,0,2).contiguous().view(sz_b,-1)
            # target_output
            target_feature, (target_hn, cn) = self.seq_lstm(target_feature)
            # hn shape is [2,Batch,hidden_size]
            sz_b = target_feature.shape[0]
            target_hn = target_hn.permute(1,0,2).contiguous().view(sz_b,-1)
            source_feature_last = source_hn
            target_feature_last = target_hn
        else:
            if self.pool_flag:
                #计算source
                pe = self.get_position_encoding(source_feature)
                source_feature = source_feature + pe
                source_feature = self.seq_transformer(source_feature,source_mask)
                source_feature_last = source_feature.mean(dim=1)
                #计算target
                pe = self.get_position_encoding(target_feature)
                target_feature = target_feature + pe
                target_feature = self.seq_transformer(target_feature,target_mask)
                target_feature_last = target_feature.mean(dim=1)
            else:
                #计算source
                cls_tokens = self.cls_token.repeat(source_shape[0],1,1)
                source_feature = torch.cat((cls_tokens, source_feature), dim=1)
                cls_mask = self.cls_mask.repeat(source_shape[0],1).to(source_feature.device)
                source_mask = torch.cat((cls_mask, source_mask), dim=1)
                pe = self.get_position_encoding(source_feature)
                source_feature = source_feature + pe
                source_feature = self.seq_transformer(source_feature,source_mask)
                source_feature_last = source_feature[:, 0, :]
                #计算target
                cls_tokens = self.cls_token.repeat(target_shape[0],1,1)
                target_feature = torch.cat((cls_tokens, target_feature), dim=1)
                cls_mask = self.cls_mask.repeat(target_shape[0],1).to(target_feature.device)
                target_mask = torch.cat((cls_mask, target_mask), dim=1)
                pe = self.get_position_encoding(target_feature)
                target_feature = target_feature + pe
                target_feature = self.seq_transformer(target_feature,target_mask)
                target_feature_last = target_feature[:, 0, :]
        
        transfer_loss = self.adapt_loss(source_feature_last, target_feature_last)
        # input classifier
        source_feature_last = self.classifier(source_feature_last)
        source_feature_last = source_feature_last.squeeze(-1)
        #calculate clf loss
        if self.n_class > 2:
            clf_loss = self.criterion(source_feature_last,source_label)
        else:
            source_feature_last = torch.sigmoid(source_feature_last)
            clf_loss = self.criterion(source_feature_last,source_label)
        
        return clf_loss,transfer_loss
    
    def predict(self,x,mask,attn_show = False):
        shape = x.shape
        #down 3D [batch*seq_len,c,f] 
        x = x.reshape(-1,shape[2],shape[3])
        # x_index = torch.where(mask.reshape(-1)!=0)[0]
        #f down 2D [batch*seq_len,hidden_dim] 
        if self.aflag:
            if attn_show:
                logics,f,a = self.net(x,attn_show)
            else:
                logics,f = self.net(x)
        else:
            logics,f = self.net(x)
        
        #up 3D [batch,seq_len,c,hidden_dim]
        f = f.reshape(shape[0],shape[1],-1)
        if self.Lstm_flag:         
            f, (f_hn, cn) = self.seq_lstm(f)
            sz_b = f.shape[0]
            f_hn = f_hn.permute(1,0,2).contiguous().view(sz_b,-1)
            f_last = f_hn
        else:
            if self.pool_flag:
                if attn_show:
                    pe = self.get_position_encoding(f)
                    f = f + pe
                    f,sa = self.seq_transformer(f,mask,attn_show=attn_show)
                else:
                    pe = self.get_position_encoding(f)
                    f = f + pe
                    f = self.seq_transformer(f,mask,mode=1)
                f_last = f.mean(dim=1)
            else:
                cls_tokens = self.cls_token.repeat(shape[0],1,1)
                f = torch.cat((cls_tokens, f), dim=1)
                cls_mask = self.cls_mask.repeat(shape[0],1).to(f.device)
                mask = torch.cat((cls_mask, mask), dim=1)
                if attn_show:
                    pe = self.get_position_encoding(f)
                    f = f + pe
                    f,sa = self.seq_transformer(f,mask,attn_show=attn_show)
                else:
                    pe = self.get_position_encoding(f)
                    f = f + pe
                    f = self.seq_transformer(f,mask)
                f_last = f[:, 0, :]
        # input classifier
        f_last = self.classifier(f_last)
        if self.n_class > 2:
            pass
        else:
            f_last = torch.sigmoid(f_last)
        
        f_last = f_last.squeeze(-1)
        if attn_show:
            if self.Lstm_flag:   
                attn = {'second_a':a}
            else:
                attn = {'second_a':a,'trial_a':sa}
            return f_last,attn
        else:
            return f_last
              
    def get_parameters(self, ratio = 1.0,initial_lr=1.0):
        params = [
            {'params': self.net.parameters(), 'lr': 1.0 * initial_lr},
            {'params': self.classifier.parameters(), 'lr': 1.0 * initial_lr},
        ]
        
        params.append(
            {'params': self.adapt_loss.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
        )
        
        # seq transformer
        if self.Lstm_flag:
             params.append(
            {'params': self.seq_lstm.parameters(), 'lr': 1.0 * initial_lr}
        )
        else:
            params.append(
                {'params': self.seq_transformer.parameters(), 'lr': ratio * initial_lr}
            )
            
            if self.pool_flag:
                 params.append(
                {'params': self.cls_token, 'lr': 1.0 * initial_lr}
            )
        return params
    
    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0).to(x.device)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.hidden_dim % 2))
        signal = signal.view(1, max_length, self.hidden_dim)
        return signal.to(x.device)
                           
if __name__ == "__main__":
    x = torch.randn(3,63,4)
    net_WA  = Feature_Extractors_WA(4,3,4,4,2)
    y = net_WA(x)
    print(y.shape)
    x = x.reshape(3,-1)
    net_WOA = Feature_Extractors_WOA(63*4,64,3)
    y2 = net_WOA(x)
    print(y2.shape)
    x = x.reshape(3,63,4)
    kwargs_a = {'input_dim':4,'n_head':3,'d_k':4,'d_v':4,'n_layers':2,'hidden_dim':64}
    net_dann = second_base_model(True,63,3,**kwargs_a)
    logics,fea = net_dann(x)
    print(logics.shape,fea.shape)
    kwargs_woa = {'input_dim':63*4,'n_layers':2,'hidden_dim':64}
    net_dann = second_base_model(False,63,3,**kwargs_woa)
    logics,fea = net_dann(x)
    print(logics.shape,fea.shape)
    
    ##trasfer
    source = torch.randn(3,63,4)
    target = torch.randn(3,63,4)
    source_label = torch.tensor([0,1,2])
    net = trasfernet_second(False,63,3,**kwargs_woa)
    cl,tl = net(source,target,source_label)
    print(cl,tl)
    para = net.get_parameters()
#    print(para)
    
    ## test cuda
    source = source.cuda()
    target = target.cuda()
    source_label = source_label.cuda()
    net.cuda()
    cl,tl = net(source,target,source_label)
    print(cl,tl)
    #test backward
    para = net.get_parameters(initial_lr=1e-3)
#    print(para)
    loss = cl + tl
    optimizer = torch.optim.SGD(para, lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #test get attn
    lo = net.predict(source)
    print(lo.shape)
    net = trasfernet_second(True,63,3,**kwargs_a)   
    net.cuda()
    lo,a = net.predict(source,True)
    print(len(a))
    cl,tl = net(source,target,source_label)
    print(cl,tl)
    
    #test trial forward
    source = torch.randn(3,800,63,4)
    target = torch.randn(3,800,63,4)
    source_label = torch.tensor([0,1,2])
    kwargs_a = {'input_dim':4,'n_head':3,'d_k':4,'d_v':4,'n_layers':2,'hidden_dim':64,'dropout':0.1}
    net_pool = trasfernet_trial(True,63,3,3,1,False,**kwargs_a)
    net_cls = trasfernet_trial(True,63,3,3,1,True,**kwargs_a)
    source = source.cuda()
    target = target.cuda()
    source_label = source_label.cuda()
    net_pool.cuda()
    net_cls.cuda()
    source_mask = torch.zeros(3,800)
    m = torch.ones(3,600)
    source_mask[:,:600] = m
    
    cl,tl = net_pool(source,target,source_label,source_mask)
    print(cl,tl)
    cl,tl = net_cls(source,target,source_label,source_mask)
    print(cl,tl)
    lo,a = net_pool.predict(source,source_mask,True)
    print(len(a))
    # test trial backward
    para = net_cls.get_parameters(0.3,initial_lr=1e-3)
#    print(para)
    loss = cl + tl
    optimizer = torch.optim.SGD(para, lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
        