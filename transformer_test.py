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

#------  encode编码器核心层 -------------
class EncodeerLayer(nn.Module):
    def __init__(self,d_model,num_heds,d_ff,dropout):
        super(EncodeerLayer,self).__init__()
        self.self_attn = MultiHeadAttention()