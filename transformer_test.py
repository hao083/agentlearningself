'''
初步模拟transformer架构
该架构提出attention机制，输入token可以随意查询其余所有的输入token，这叫做自注意力机制
通过自注意力机制，可以快速判断出语义，例如“它”这个代词，开始的词向量只有基础信息，无法得出这个代词到底指代什么
自注意力机制可以帮助解决这一问题
自注意力机制只能简单的将相关信息（kqv）加入到当前token，但是要真正确认语义，还需要神经网络的进一步计算得出
例如“我购买了一部黑色的旗舰高价苹果”，最开始“苹果”这一个词的词向量只包含水果信息，通过自注意力机制，可以得出它具有高价和科技属性
但是到底是那几个属性占据大头呢，这就需要神经网络来计算解决。FNN得出科技和高价属性，并通过权重计算，将“苹果”大概率指向为“手机科技产品”，而不是“农作物水果”
最后，解码器在工作时，回顾解码器已经产出的token和编码器中的token信息，这就是交叉注意力机制
'''

import torch
import torch.nn as nn
import math

#------ 占位符模块 ----------
class PositionalEncoding(nn.Module):
    '''
    位置编码模块
    '''
    def forward(self,x):
        pass

class MultiHeadAttention(nn.Module):
    '''
    多头注意力机制模块
    '''
    def forward(self,query,key,value,mask):
        pass

class PositionWiseFeedFrward(nn.Module):
    '''
    位置前馈网络模块
    '''
    def forward(self,x):
        pass

#------  encoder编码器核心层 -------------
class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heds,d_ff,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn = MultiHeadAttention()#多头注意力机制
        self.feed_forward = PositionWiseFeedFrward()#前馈神经网络模块
        self.noam1 = nn.LayerNorm(d_model)
        self.noam2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        #残差分析、残差连接与层归一化后续在进行补充实现
        #1、多头注意力机制
        attn_output = self.self_attn(x,x,x,mask)
        x = self.norm1(x + self.dropout(attn_output))

        #2、前馈神经网络FNN
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

#------  decoder解码器核心层  --------------
class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super(DecoderLayer,self).__init__()
        self.self_attn = MultiHeadAttention()#多头注意力机制
        self.cross_attn = MultiHeadAttention()#交叉注意力机制
        self.feed_forward = PositionWiseFeedFrward()#FNN
        self.norm1 = nn.Module(d_model)
        self.norm2 = nn.Module(d_model)
        self.norm3 = nn.Module(d_model)
        self.dropout = nn.Dropout(d_model)

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        #1、掩码多头注意力机制，主要是解码器对自身的限制，防止抄答案
        attn_output = self.self_attn(x,x,x,tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        #2、交叉注意力机制，主要是借助编码器输出来学习得出解码器的输出
        cross_attn_output = self.cross_attn(x,encoder_output,encoder_output,src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        #3、前馈神经网络学习，得出最终输出
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x




