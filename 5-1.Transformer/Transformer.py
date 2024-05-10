"""
# File        : Transformer.py
# Author      : Shard Zhang
# Date        : 2024/05/09 15:16
# Brief       : Transformer模型
# https://github.com/simidagogogo/nlp-tutorial/blob/master/5-1.Transformer/Transformer.py
#
# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# source_vocab_size: 原始的语言词表大小, 可以非常大
# target_vocab_size: 翻译后语言词表大小, 可以非常大
# tgt_seq_len: decoder侧(原始的语言)句子长度, 可能很短
# src_seq_len: encoder侧(翻译后语言)句子长度, tgt_seq_len可能不等于src_seq_len

# S: Symbol that shows starting of decoding input
# E: Symbol that shows Ending of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is shorter than time steps

def make_batch(sentences):
    """
    数据预处理, 对原始句子进行text to sequence编码
    """
    # [batch_size, sen_len]
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    print(f"input_batch: {input_batch}")    # [[1, 2, 3, 4, 0]]
    print(f"output_batch: {output_batch}")  # [[5, 1, 2, 3, 4]]
    print(f"target_batch: {target_batch}")  # [[1, 2, 3, 4, 6]]

    return (torch.LongTensor(input_batch),
            torch.LongTensor(output_batch),
            torch.LongTensor(target_batch))


def get_sinusoid_encoding_table(n_position, d_model):
    """
    正弦编码是一种常见的位置编码方法，它利用正弦函数的周期性来编码位置信息
    :param n_position:
    :param d_model:
    :return:
    """
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    print(f"sinusoid_table : {sinusoid_table}, size: {sinusoid_table.shape}") # (n_position, d_model)

    return torch.FloatTensor(sinusoid_table)


def get_attn_pad_mask(seq_q, seq_k):
    """
    attention中两个mask之一, 另一个是get_attn_sequent_mask()

    为什么需要pad_mask掩码?
    序列往往包含"填充元素"（padding）以保持固定长度. 但是这些元素不包含有用的信息, 在计算注意力权重时需要被屏蔽掉(忽略zero padding词)
    @seq_q : [batch_size, len_q]
    @seq_k : [batch_size, len_k]. 对于decoder, 这里是encoder_input
    @:return: [batch_size, len_q, len_k]
    """

    # 这两个序列通常来自同一个输入序列，但可能经过不同的处理
    _, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token
    # pad_mask: [batch_size, 1, len_k(=len_q)], one is masking
    pad_mask = seq_k.data.eq(0).unsqueeze(1)

    # [batch_size, len_q, len_k]
    # TODO: 这里为什么可以使用expand呢? 可以使用repeat()么?
    return pad_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequent_mask(seq):
    """
    attention中两个mask之一, 另一个是get_attn_pad_mask()
    仅用于Decoder的Masked Multi-Head Self-attention中.

    subsequent mask掩码用于处理序列数据, 它会与自注意力层的输出相乘, 以屏蔽序列中后续位置, 以确保解码器只能使用之前的输出来预测下一个输出
    序列生成任务中(如自回归模型)，模型在预测当前词时不应该使用未来信息(即未来词), 从而让模型正确学习序列元素之间的依赖关系，同时保持预测的一致性

    @seq: [batch_size, tgt_seq_len]
    @return: [batch_size, tgt_seq_len, tgt_seq_len]
    """

    # [batch_size, tgt_seq_len, tgt_seq_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]

    # 创建subsequent mask(上三角矩阵): 对角线以上的元素为1(不包括对角线)，其余元素都为0
    # one is masking
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)

    # 将NumPy数组转换为PyTorch张量，因为后续操作需要在计算图中进行
    # subsequent_mask = torch.ByteTensor(subsequent_mask)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()

    # [batch_size, tgt_seq_len, tgt_seq_len]
    return subsequent_mask


def showgraph(attn, text):
    """
    最后一层的 第一个head的 注意力权重的可视化
    :param attn: List[[batch_size, heads, len_q, len_k]]
    """

    # 有多少个Block堆叠层
    print(f"Step0. attn: {len(attn)}") # 6

    # attn[-1] 获取最后一层的注意力权重
    print(f"Step1. attn: {attn[-1].shape}") # torch.Size([1, 8, 5, 5])

    # .squeeze(0) 去掉batch_size维度(因为这里仅有一条样本)
    print(f"Step2. attn: {attn[-1].squeeze(0).shape}") # torch.Size([8, 5, 5])

    # [0]选择第一个头部的注意力权重
    print(f"Step3. attn: {attn[-1].squeeze(0)[0].shape}") # torch.Size([5, 5])

    # 这里的sequeeze(0)已经没有影响, 加不加都行
    print(f"Step4. attn: {attn[-1].squeeze(0)[0].squeeze(0).shape}") # torch.Size([5, 5])

    # 将数据类型被转换为NumPy数组，以便可以被matplotlib正确处理
    attn = attn[-1].squeeze(0)[0].data.numpy()

    # [[0.24999687 0.24999605 0.25000238 0.25000474 0.        ]
    #  [0.24999689 0.24999605 0.25000238 0.25000474 0.        ]
    #  [0.24999687 0.24999605 0.2500024  0.25000474 0.        ]
    #  [0.24999686 0.24999607 0.25000244 0.2500047  0.        ]
    #  [0.24999686 0.24999607 0.2500024  0.2500047  0.        ]]
    # torch.Size([5, 5])
    print(f"attn: {attn}")

    # figsize参数定义了图形的大小，这里使用n_heads作为宽和高的尺寸，意味着图形的大小与注意力头的数量相匹配
    fig = plt.figure(figsize=(n_heads, n_heads))  # [n_heads, n_heads]

    # 在创建的图形上添加一个子图(在1行1列的图形中添加第一个子图)
    ax = fig.add_subplot(1, 1, 1)

    # 使用matshow函数在子图上显示注意力矩阵
    # 'viridis'是matplotlib中预定义的颜色映射之一，它是一种连续的颜色渐变，从黄绿色渐变到深蓝色，非常适合用来表示数据的渐变
    ax.matshow(attn, cmap='viridis')

    # ['', 'ich', 'mochte', 'ein', 'bier', 'P']
    # 在最左侧添加一个额外的空白标签(序列第一个元素), 以提供一个占位符以作为视觉分隔符或对齐
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 10}, rotation=0)
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 10})

    plt.xlabel("encoder input")
    plt.ylabel("label")
    plt.title(text)

    plt.show()



def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1.
    This is necessary for inference as we don't know the target sequence input.
    Therefore we try to generate the target input word by word, then feed it into the transformer.

    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding

    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 5
    :return: The target input sequence
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)

    # 初始化解码器的输入, 特别是在序列生成任务中, 比如机器翻译或文本摘要
    #   1) 序列生成的起始点：在序列生成任务中，解码器的初始输入通常是全零的张量，表示序列的开始
    #   2) 占位符：在某些模型中，可能需要一个初始的输入张量作为循环的起点，稍后会根据模型的输出动态更新这个张量
    dec_input = torch.zeros(1, tgt_seq_len).type_as(enc_input.data)

    next_symbol = start_symbol
    for i in range(0, tgt_seq_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        print(f"debug. projected.shape: {projected.shape}") # torch.Size([1, 50, 7])

        # 对Decoder输出进行投影, 以获得下一个单词的概率分布
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        print(f"debug. prob.shape: {prob.shape}") # torch.Size([50])

        next_word_idx = prob.data[i].item()
        print(f"debug. i: {i}, next_word_idx: {next_word_idx}, next_word: {number_dict[next_word_idx]}")
        next_symbol = next_word_idx
    return dec_input


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # Part1. encoder
        self.encoder = Encoder()

        # Part2. encoder
        self.decoder = Decoder()

        # Part3. Linear Layer
        # 将d_model映射到tgt_vocab_size, 以进行多分类任务
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)    # (d_model, tgt_vocab_size)

    def forward(self, enc_inputs, dec_inputs):
        """
        @enc_inputs:  (batch_size, src_seq_len). 原始的text to sequence编码, 形如 [[1, 2, 3, 4, 0]]
        @dec_inputs: (batch_size, tgt_seq_len). 原始的text to sequence编码, 形如 [[5, 1, 2, 3, 4]]
        @return:
            dec_logits: [batch_size * tgt_seq_len, tgt_vocab_size]
            enc_self_attns: List[[batch_size, n_heads, src_seq_len, src_seq_len]]
            dec_self_attns: List[[batch_size, n_heads, tgt_seq_len, tgt_seq_len]]
            dec_enc_attns:  List[[batch_size, n_heads, tgt_seq_len, src_seq_len]]
        """

        # Part1. encoder
        # enc_outputs: [batch_size, src_seq_len, d_model]
        # enc_self_attns: List[[batch_size, n_heads, src_seq_len, src_seq_len]]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # Part2. decoder
        # dec_outputs: [batch_size, tgt_seq_len, d_model]
        # dec_self_attns: List[[batch_size, n_heads, tgt_seq_len, tgt_seq_len]]
        # dec_enc_attns:  List[[batch_size, n_heads, tgt_seq_len, src_seq_len]]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        print(f"debug. dec_inputs: {dec_inputs.shape}") # (1, 5)
        print(f"debug: dec_outputs.shape: {dec_outputs.shape}") # torch.Size([1, 5, 512])

        # Part3. Linear
        # dec_logits : [batch_size, tgt_seq_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        print(f"debug: dec_logits.shape: {dec_logits.shape}") # torch.Size([1, 5, 9])

        # dec_logits: [batch_size * tgt_seq_len, tgt_vocab_size]
        return (dec_logits.view(-1, dec_logits.size(-1)),
                enc_self_attns,
                dec_self_attns,
                dec_enc_attns)


class Encoder(nn.Module):
    """
    Transformer编码器Encoder, 将词序列的one-hot表示编码为emb表示
    """
    def __init__(self):
        super(Encoder, self).__init__()

        # 词表emb矩阵, [src_vocab_size, d_model]
        self.src_emb = nn.Embedding(src_vocab_size, d_model)

        # 输入词序列的位置编码矩阵, [src_seq_len, d_model]
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(src_seq_len, d_model),
            freeze=True)

        # N个EncoderBlock的堆叠
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])


    def forward(self, enc_inputs):
        """
        @enc_inputs : [batch_size, src_seq_len]. 原始的one-hot编码结果
        :return  [batch_size, src_seq_len, d_model]
        """

        # 1. 计算自注意力掩码
        # enc_self_attn_mask: [batch_size, len_q, len_k]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        # 2. 计算encoder层输入
        # enc_outputs: [batch_size, src_seq_len, d_model]
        enc_inputs = self.src_emb(enc_inputs) + self.pos_emb(enc_inputs)

        # 3. 计算经过各个EncoderBlock
        # 用于记录attention的相关系数, 用于可视化
        # List[[batch_size, n_heads, len_q, len_k]]
        enc_self_attns = []

        enc_outputs = None
        for layer in self.layers:
            # 3.1. 将输入喂入每层EncoderBlock, 得到输出
            # enc_outputs: [batch_size, src_seq_len, d_model]
            # enc_self_attn: [batch_size, n_heads, len_q, len_k]
            enc_outputs, enc_self_attn = layer(enc_inputs, enc_self_attn_mask)

            # 3.2. 记录每一层的encoder自注意力掩码
            enc_self_attns.append(enc_self_attn)

            # 3.3. 迭代. 将输出作为输入
            enc_inputs = enc_outputs
        return enc_outputs, enc_self_attns



class EncoderLayer(nn.Module):
    """
    EncoderBlock, Encoder堆叠单元
    """
    def __init__(self):
        super(EncoderLayer, self).__init__()

        # Part1. Self-Attention Sublayer
        self.encoder_self_attn = MultiHeadAttention()

        # Part2. FFN Sublayer
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        @enc_inputs : [batch_size, src_seq_len, d_model]
        @enc_self_attn_mask: [batch_size, len_query, len_key]
        """

        # Part1. Self Attention Sublayer
        # enc_inputs to same Q,K,V
        # enc_outputs: [batch_size, src_seq_len, d_model]
        enc_outputs, attn = self.encoder_self_attn(enc_inputs,
                                                   enc_inputs,
                                                   enc_inputs,
                                                   enc_self_attn_mask)

        # Part2. FFN Sublaer
        # enc_outputs: [batch_size, src_seq_len, d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention中包括了 MultiHeadAttention + Add&Norm
    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        # 将d_model映射到多个头的d_k空间
        self.W_Q = nn.Linear(d_model, d_k * n_heads) # [d_model, d_k * n_heads]
        self.W_K = nn.Linear(d_model, d_k * n_heads) # [d_model, d_k * n_heads]
        self.W_V = nn.Linear(d_model, d_v * n_heads) # [d_model, d_v * n_heads]

        # 将多个头的d_k空间隐射回d_model空间
        self.linear = nn.Linear(n_heads * d_v, d_model)

        # LN层
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        """
        # Q: [batch_size, len_q, d_model]
        # K: [batch_size, len_k, d_model]
        # V: [batch_size, len_v, d_model]
        # attn_mask: [batch_size, len_q, len_k]

        # 对于Decoder的互注意力SubLayer.
            Q: [batch_size, tgt_seq_len, d_model]
            K: [batch_size, src_seq_len, d_model]
            V: [batch_size, src_seq_len, d_model]
            attn_mask: [batch_size, tgt_seq_len, src_seq_len]
        """

        # 备份输入Q, 用于后续残差的输入之一
        residual = Q

        # 获取batch_size
        batch_size = Q.size(0)

        # Step1. 将d_model映射到多个头的d_k空间
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # ==================================================================
        # q_s: [batch_size, n_heads, len_q, d_k]
        # 对于Decoder的互注意力SubLayer. [batch_size, n_heads, tgt_seq_len, d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # k_s: [batch_size, n_heads, len_k, d_k]
        # 对于Decoder的互注意力SubLayer. [batch_size, n_heads, src_seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # v_s: [batch_size, n_heads, len_v, d_v]
        # 对于Decoder的互注意力SubLayer. [batch_size, n_heads, src_seq_len, d_v]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        # ==================================================================

        # attn_mask : [batch_size, n_heads, len_q, len_k]
        # 对于Decoder的互注意力SubLayer. [batch_size, n_heads, tgt_seq_len, src_seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        # 对于Decoder的互注意力SubLayer.
        #   context: [batch_size, n_heads, tgt_seq_len, d_v]
        #   attn:    [batch_size, n_heads, tgt_seq_len, src_seq_len]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # Step2. 将多个head的输出concat到一起
        # context: [batch_size, len_q, n_heads * d_v]
        # 由于.transpose()操作可能会产生非连续张量, .contiguous()确保数据连续性, 是.view()操作的前提条件, 使得.view()能够正确执行
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)

        # Step3. 将多个头的d_k空间重新映射回d_model
        # output: [batch_size, len_q, d_model]
        # 对于Decoder的互注意力SubLayer. [batch_size, tgt_seq_len, d_model]
        output = self.linear(context)

        # Step4. Add&Norm
        return self.layer_norm(output + residual), attn


class ScaledDotProductAttention(nn.Module):
    """
    点乘(内积)的Attention
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v, d_v]
        attn_mask: [batch_size, n_heads, len_q, len_k]
        @:return context: 经过Attention加权之后的V
        @:return attn: Q和K计算所得权重值的Softmax归一化概率值
        """

        # Step1. 计算score
        # [batch_size, n_heads, d_k, len_k]
        K_transponse = K.transpose(-1, -2)

        # scores : [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        scores = torch.matmul(Q, K_transponse) / np.sqrt(d_k)

        # Step2. 填充大负数
        # Fills elements of self tensor with value where mask is one.
        scores.masked_fill_(attn_mask, -1e9) # 将掩码中值为1(padding)的位置的分数替换为一个非常大负数

        # Step3. Softmax归一化(权重值)
        # attn: [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)

        # Step4. 计算Attention后的V
        # [batch_size, n_heads, len_q, len_k] * [batch_size, n_heads, len_v, d_v] = [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)

        # len_q可能不等于len_k.
        #   对于encoder, len_q = len_k = len_v = src_seq_len
        #   对于decoder, len_q = tgt_seq_len, len_k = len_v = src_seq_len

        # context和V的shape是否一定一致?
        #   对于encoder, len_q = len_k = len_v = src_seq_len, context和V的shape一致
        #   对于encoder, len_q = tgt_seq_len, len_k = len_v = src_seq_len, context和V的shape不一致

        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        return context, attn


class PoswiseFeedForwardNet(nn.Module):
    """
    pos-wise 前馈网络
    todo: 为什么要使用卷积层Conv1d(), 而不是直接使用全连接层Linear()
    """

    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()

        # kernel_size=1卷积核只覆盖序列中单个元素, “逐点卷积”（pointwise convolution), 不会捕捉序列中元素之间的依赖关系, 而是对每个单独的元素应用一个线性变换
        # 第一层卷积(全连接层)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        # LN层
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        通过使用较小的卷积核和较大的输出通道数，模型可以在不改变序列长度的情况下，学习更丰富的特征表示
        此外，由于卷积核的大小为1，这种操作实际上是一个全连接层，它对序列中的每个位置独立地应用相同的变换，这有助于保持位置不变性
        # inputs : [batch_size, len_q, d_model]
        """

        # Step1. 备份输入, 后续用于残差输入
        # [batch_size, len_q, d_model]
        residual = inputs

        # Step2. 经过一层全连接层
        # 卷积层期望输入的形状为(batch_size, in_channels, sequence_length)
        # 对于一维卷积来说，in_channels通常对应于特征的数量d_model
        # [batch_size, d_model, len_q] 经过卷积 [d_model, d_ff] = [batch_size, d_ff, len_q]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))

        # [batch_size, d_ff, len_q] 经过卷积 [d_ff, d_model] = [batch_size, d_model, len_q]
        output = self.conv2(output)

        # [batch_size, len_q, d_model]
        output = output.transpose(1, 2)

        # Step3. 经过LN层
        # output: [batch_size, len_q, d_model]
        return self.layer_norm(output + residual)


class Decoder(nn.Module):
    """
    Transformer的Decoder
    """
    def __init__(self):
        super(Decoder, self).__init__()

        # 词表emb矩阵, [tgt_vocab_size, d_model]
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)

        # 输入词序列的位置编码矩阵, [tgt_seq_len, d_model]
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(tgt_seq_len + 1, d_model),
            freeze=True)

        # DecoderBlock堆叠层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        @dec_inputs : [batch_size, tgt_seq_len]
        @enc_inputs:  [batch_size, src_seq_len]
        @enc_outputs: [batch_size, src_seq_len, d_model]
        @return:
        """

        # Step1. decoder的输入(emb + 位置编码)
        # dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5, 1, 2, 3, 4]]))
        # dec_outputs: [batch_size, tgt_seq_len, d_model]
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(dec_inputs)


        # Step2. decoder中的自注意力层的mask(decoder这里有两个mask)
        # Step2.1. pad mask
        # dec_self_attn_pad_mask: [batch_size, tgt_seq_len, tgt_seq_len]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

        # Step2.2. subsequent mask
        # dec_self_attn_subsequent_mask: [batch_size, tgt_seq_len, tgt_seq_len]
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        # Step2.3. 两者相加
        # dec_self_attn_mask: [batch_size, tgt_seq_len, tgt_seq_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)


        # Step3. 交互自注意力层的mask(enc_inputs仅仅用于获取mask的形状)
        # dec_enc_attn_mask: [batch_size, tgt_seq_len, src_seq_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)


        # Step4. DecoderBlock堆叠层
        # 列表的长度就是decoder的层数
        dec_self_attns = []
        dec_enc_attns = []
        for layer in self.layers:
            # dec_outputs  : [batch_size, tgt_seq_len, d_model]
            # dec_self_attn: [batch_size, n_heads, tgt_seq_len, tgt_seq_len]
            # dec_enc_attn : [batch_size, n_heads, tgt_seq_len, src_seq_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs,       # decoder输入  (每层DecodeBlock不同)
                                                             enc_outputs,       # encoder输入  (每层DecodeBlock相同)
                                                             dec_self_attn_mask,# 自注意力层mask(每层DecodeBlock相同)
                                                             dec_enc_attn_mask) # 互注意力层mask(每层DecodeBlock相同)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        # dec_outputs: [batch_size, tgt_seq_len, d_model]
        # dec_self_attns: List[[batch_size, n_heads, tgt_seq_len, tgt_seq_len]]
        # dec_enc_attns:  List[[batch_size, n_heads, tgt_seq_len, src_seq_len]]
        return dec_outputs, dec_self_attns, dec_enc_attns


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

        # Part1. self-attention 自注意力层
        self.dec_self_attn = MultiHeadAttention()

        # Part2. attention 交互注意力层
        self.dec_enc_attn = MultiHeadAttention()

        # Part3. FFN
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        @dec_inputs: 解码器的输入,  [batch_size, tgt_seq_len, d_model]
        @enc_outputs: 编码器的输出, [batch_size, src_seq_len, d_model]
        @dec_self_attn_mask: 解码器的 自注意力mask, [batch_size, tgt_seq_len, tgt_seq_len]. padding_mask + subsequent_mask两者相加
        @dec_enc_attn_mask : 解码器的 互注意力mask, [batch_size, tgt_seq_len, src_seq_len]. 仅包含padding_mask
        """

        # Part1. 自注意力层. Q, K和V来自decoder
        # dec_outputs: [batch_size, tgt_seq_len, d_modal]
        # dec_self_attn: [batch_size, n_heads, tgt_seq_len, tgt_seq_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs,  # Q, [batch_size, tgt_seq_len, d_model]
                                                        dec_inputs,  # K, [batch_size, tgt_seq_len, d_model]
                                                        dec_inputs,  # V, [batch_size, tgt_seq_len, d_model]
                                                        dec_self_attn_mask  # [batch_size, tgt_seq_len, tgt_seq_len]
                                                        )

        # Part2. 交互注意力层. Q来自decoder, K和V来自encoder
        # dec_outputs: [batch_size, tgt_seq_len, d_modal]
        # dec_enc_attn: [batch_size, n_heads, tgt_seq_len, src_seq_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs,  # Q, [batch_size, tgt_seq_len, d_modal]
                                                      enc_outputs,  # K, [batch_size, src_seq_len, d_model]
                                                      enc_outputs,  # V, [batch_size, src_seq_len, d_model]
                                                      dec_enc_attn_mask # [batch_size, tgt_seq_len, src_seq_len]
                                                      )

        # Part3. FFN层
        # dec_outputs: [batch_size, tgt_seq_len, d_model]
        dec_outputs = self.pos_ffn(dec_outputs)

        # dec_outputs: [batch_size, tgt_seq_len, d_model]
        # dec_self_attn: [batch_size, n_heads, tgt_seq_len, tgt_seq_len]
        # dec_enc_attn: [batch_size, n_heads, tgt_seq_len, src_seq_len]
        return dec_outputs, dec_self_attn, dec_enc_attn


if __name__ == '__main__':
    # Note: 这里仅为一条样本(由三句话组成)
    sentences = [
        'ich mochte ein bier P',    # encoder input
        'S i want a beer',          # decoder input
        'i want a beer E'           # label
    ]

    # Transformer Parameters
    # 输入词表(词典)
    # word -> idx
    src_vocab = {
        'P': 0, # Padding Should be Zero, eq(zero) is PAD token
        'ich': 1,
        'mochte': 2,
        'ein': 3,
        'bier': 4,
        # 以下为自己添加
        'poer': 5,
        'tite': 6
    }
    # 输入词表大小
    src_vocab_size = len(src_vocab) # 7

    # 输出词表(词典)
    # word -> idx
    tgt_vocab = {
        'P': 0, # Padding Should be Zero, eq(zero) is PAD token
        'i': 1,
        'want': 2,
        'a': 3,
        'beer': 4,
        'S': 5,
        'E': 6,
         # 以下为自己添加
        'A': 7,
        'B': 8
    }
    # 输出词表大小
    tgt_vocab_size = len(tgt_vocab) # 9
    # idx -> word
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}

    # 最大长度为5, 长度不够的句子做padding.
    # 目的是对齐batch_size中每条样本的长度, 从而形成矩阵加速计算.
    src_seq_len = 5  # length of source
    tgt_seq_len = 5  # length of target

    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_heads = 8  # number of heads in Multi-Head Attention
    n_layers = 6  # number of Encoder of Decoder Layer

    # 实例化模型
    model = Transformer()

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 输入数据
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    # 以下为标准流程
    for epoch in range(10):
        # 1. 清空梯度
        optimizer.zero_grad()

        # outputs: [batch_size, tgt_vocab_size]
        # input_batch: [[1, 2, 3, 4, 0]]
        # output_batch: [[5, 1, 2, 3, 4]]
        # target_batch: [[1, 2, 3, 4, 6]]

        # 2. 前向传播
        # outputs: [batch_size * tgt_seq_len, tgt_vocab_size]
        # enc_self_attns: List[[batch_size, heads, src_seq_len, src_seq_len]]
        # dec_self_attns: List[[batch_size, heads, tgt_seq_len, tgt_seq_len]]
        # dec_enc_attns : List[[batch_size, heads, tgt_seq_len, src_seq_len]]
        dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        print(f"debug. outputs.shape: {dec_logits.shape}") # torch.Size([1 * 5, 7])
        print(f"debug. len(enc_self_attns): {len(enc_self_attns)}") # 6, 表示有6个Block堆叠层
        print(f"debug. len(dec_self_attns): {len(dec_self_attns)}") # 6, 表示有6个Block堆叠层
        print(f"debug. len(dec_enc_attns): {len(dec_enc_attns)}")   # 6, 表示有6个Block堆叠层

        # labels: [batch_size * tgt_seq_len, ]
        labels = target_batch.contiguous().view(-1)

        # 3. 计算损失
        # 为什么需要先调用.contiguous()? .view()要求输入的张量必须是连续的
        loss = criterion(dec_logits, labels)
        print(f"Epoch: {(epoch + 1):04d}, cost: {loss:.6f}")

        # 4. 反向传播, 计算梯度
        loss.backward()

        # 5. 更新模型参数
        optimizer.step()

    # Eval
    print('==== decoder inputs ====')
    predict_logit, _, _, _ = model(enc_inputs, dec_inputs)
    # .data 获取张量中的实际数值数据（忽略梯度信息）
    # .max 函数返回一个元组，其中第一个元素是最大值，第二个元素是最大值的索引
    predict_logit = predict_logit.data.max(1, keepdim=True)[1]
    # ich mochte ein bier P -> ['i', 'beer', 'i', 'beer', 'want']
    print(sentences[0], '->', [number_dict[n.item()] for n in predict_logit.squeeze()])

    print('==== greedy decoder inputs ====')
    greedy_dec_input = greedy_decoder(model, enc_inputs, start_symbol=tgt_vocab["S"])
    predict_logit_greedy, _, _, _ = model(enc_inputs, greedy_dec_input)
    predict_logit_greedy = predict_logit_greedy.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict_logit_greedy.squeeze()])

    # print('==== first head of last state enc_self_attns ====')
    # showgraph(enc_self_attns, "first head of last state enc_self_attns")
    #
    # print('==== first head of last state dec_self_attns ====')
    # showgraph(dec_self_attns, "first head of last state dec_self_attns")
    #
    # print('==== first head of last state dec_enc_attns ====')
    # showgraph(dec_enc_attns, "first head of last state dec_enc_attns")