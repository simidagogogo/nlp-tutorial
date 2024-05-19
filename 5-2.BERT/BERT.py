# coding=utf-8
# ********************************************************************************
# File        : BERT.py
# Author      : Shard Zhang
# Date        : 2024-05-08 20:02
# Brief       : bert模型
# https://github.com/simidagogogo/nlp-tutorial/blob/master/5-2.BERT/BERT.py
# ********************************************************************************

# %%
# code by Tae Hwan Jung(Jeff Jung) @graykode
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#             https://github.com/JayParks/transformer,
#             https://github.com/dhlee347/pytorchic-bert

import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy


def show_words(input_ids, number_dict):
    res = []
    for i in input_ids:
        # print(f"{i} -> {number_dict[i]}")
        res.append(number_dict[i])
    return res

def make_batch(sentences, token_list, max_pred, number_dict):
    """
    sample IsNext and NotNext to be same in small batch size
    @sentences: List[str]
    @token_list: List[List[bigint]]
    @max_pred: 最大预测长度
    @number_dict: 词典index -> word
    :return:
        input_ids: 长度为maxlen的token_id序列
        segment_ids: 长度为maxlen的seg_id序列
        masked_tokens: 长度为max_pred的的token_id序列
        masked_pos: 长度为max_pred的的token_id_pos序列
        isNext: 是否下一个segment
    """
    assert batch_size % 2 == 0, "batch_size must be even"

    batch = []
    input_ids_mask_batch = []
    positive = 0
    negative = 0

    # 正负样本的比例一致, 即各为50^
    while positive != batch_size / 2 or negative != batch_size / 2:
        # sample random index(0~len(sentences)-1) in sentences
        tokens_a_index = randrange(len(sentences))
        tokens_b_index = randrange(len(sentences))

        # 随机选取两个句子
        tokens_a = token_list[tokens_a_index]
        tokens_b = token_list[tokens_b_index]

        # 构造输入的sequence ids
        # [1, 17, 19, 7, 2, 4, 25, 19, 13, 27, 28, 19, 6, 2]
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]

        # 记录mask的操作: NOTHIT/MASK/RANDOM/NOTHING/PAD
        input_ids_mask = ["NOTHIT"] * len(input_ids)

        # 由于后面会修改input_ids的值, 因此这里提前做一个备份
        input_ids_bak = copy.deepcopy(input_ids)
        print(f"input_ids_bak: {input_ids_bak}")

        # ['[CLS]', 'thanks', 'you', 'romeo', '[SEP]', 'nice', 'meet', 'you', 'too', 'how', 'are', 'you', 'today', '[SEP]']
        print(f"sequences -> {show_words(input_ids, number_dict)}")

        # segment0表示A, 包括CLS和SEP. segment1表示B, 包括SEP
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))  # 15 % of tokens in one sentence
        print(f"n_pred: {n_pred}")

        # 可以进行mask的token_id位置
        cand_maked_pos = [i for i, token in enumerate(input_ids) if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        print(f"cand_maked_pos: {cand_maked_pos}")

        # 这个shuffle有什么必要? 要随机的算出n_pred个
        shuffle(cand_maked_pos)

        # 随机选取n_pred个位置进行MASK
        masked_tokens = []
        masked_pos = []
        for pos in cand_maked_pos[:n_pred]:
            print(f"masked_pos: {pos}, masked_token: {input_ids[pos]}")
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            # 对于命中预估的15%:
            #   80%的概率做MASK,
            #   10%的概率做随机替换
            #   10%的概率不做任何变化
            if random() < 0.8:  # 80%
                input_ids[pos] = word_dict['[MASK]']  # make mask
                input_ids_mask[pos] = "MASK"
            elif random() < 0.5:  # 10%
                index = randint(0, vocab_size - 1)  # random index in vocabulary
                input_ids[pos] = word_dict[number_dict[index]]  # replace
                input_ids_mask[pos] = "RANDOM"
            else:
                # 不做任何变化
                input_ids_mask[pos] = "NOTHING"
                pass

        # Zero Paddings
        # 针对整个句子维度的padding
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        input_ids_bak.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        input_ids_mask.extend(["PAD"] * n_pad)

        # Zero Padding (100% - 15%) tokens
        # 针对masked tokens维度的padding
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)


        # 判断正负样本
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids_bak, input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
            input_ids_mask_batch.append(input_ids_mask)
            print(f"input_ids_mask: {input_ids_mask}")
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids_bak, input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
            input_ids_mask_batch.append(input_ids_mask)
            print(f"input_ids_mask: {input_ids_mask}")
            negative += 1

    print(f"fuck. input_ids_mask_batch: {input_ids_mask_batch}")
    print(f"fuck. len(input_ids_mask_batch): {len(input_ids_mask_batch)}")
    return input_ids_mask_batch, batch


# Proprecessing Finished

def get_attn_pad_mask(seq_q, seq_k):
    """
    Encoder的attention mask

    :seq_q: [batch_size, len_q]
    :seq_k: [batch_size, len_k]
    :return: [batch_size, len_q, len_k]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token,
    # pad_attn_mask: [batch_size, 1, len_k(=len_q)], one is masking
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    # [batch_size, len_q, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def gelu(x):
    """
    Implementation of the gelu activation function by Hugging Face
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    """
    token的输入emb层
    """
    def __init__(self):
        super(Embedding, self).__init__()

        # token embedding
        # [vocab_size, d_model]
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # position embedding
        # [max_len, d_model]
        self.pos_embed = nn.Embedding(maxlen, d_model)

        # segment(token type) embedding
        # [segment, d_model]
        self.seg_embed = nn.Embedding(n_segments, d_model)

        # LN层
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        """
        :x: [batch_size, seq_len], 第二维的取值为0~vocab_size-1
        :seg: [batch_size, seq_len], 第二维的取值为0,1
        """
        # 获取seqence长度
        seq_len = x.size(1)

        # 在序列模型中作为位置编码（position encoding）
        pos = torch.arange(seq_len, dtype=torch.long)

        # [seq_len,] -> [batch_size, seq_len]
        # expand 方法不会消耗额外的内存，因为它不复制数据，而是返回一个新的视图，其中某些维度被虚拟地扩展以匹配另一个张量的形状
        pos = pos.unsqueeze(0).expand_as(x)

        # 位置编码: 三者相加
        embedding = self.token_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)

        # LN层
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_k, d_v]
        # attn_mask : [batch_size, n_heads, len_q, len_k]
        :return:
            context: [batch_size, n_heads, len_q, d_v]
            atten:   [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        """
        # scores: [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        # Fills elements of self tensor with value where mask is one.
        scores.masked_fill_(attn_mask, -1e9)

        # attn: [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        attn = nn.Softmax(dim=-1)(scores)

        # context: [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制层
    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        """
        # Q:        [batch_size, len_q, d_model]
        # K:        [batch_size, len_k, d_model]
        # V:        [batch_size, len_k, d_model]
        :attn_mask: [batch_size, len_q, len_k]
        """

        # [batch_size, len_q, d_model]
        residual = Q
        batch_size = Q.size(0)

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, n_heads, len_q, d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # k_s: [batch_size, n_heads, len_k, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)

        # v_s: [batch_size, n_heads, len_k, d_v]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # attn_mask : [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q(=len_k), len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s,
                                                    k_s,
                                                    v_s,
                                                    attn_mask)

        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)

        # 将多头输出拼接后的结果, 经过线性层变换会d_model
        # output: [batch_size, len_q, d_model]
        output = nn.Linear(n_heads * d_v, d_model)(context)

        # output: [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """
    编码器层
    """
    def __init__(self):
        super(EncoderLayer, self).__init__()

        # Part1. encoder self attention
        self.enc_self_attn = MultiHeadAttention()

        # Part2. encoder feed forward
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :enc_inputs: [batch_size, len_q, d_model]
        :enc_self_attn_mask: [batch_size, len_q, len_k]
        :return:
            enc_outputs: [batch_size, len_q, d_model]
            attn:        [batch_size, len_q(=len_k), len_k(=len_q)]
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs,
                                               enc_inputs,
                                               enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        # enc_outputs: [batch_size, len_q, d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class BERT(nn.Module):
    """
    BERT.
    todo 这个模型结构还是不是很清晰, 需要自己动手画一下
    """
    def __init__(self):
        super(BERT, self).__init__()

        # Step1. input_emb
        self.embedding = Embedding()

        # Step.2 EncoderBlock堆叠层
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        # Step3.1 NSP线性层
        # 1)最后一层线性层
        self.fc1 = nn.Linear(d_model, d_model)
        # 2)双曲正切激活函数
        self.tanh = nn.Tanh()
        # 3)最后一层线性层
        self.fc2 = nn.Linear(d_model, 2)

        # Step3.2 Masked LM线性层
        # 1)线性层
        self.linear = nn.Linear(d_model, d_model)
        # 2)GELU激活函数
        self.gelu = gelu
        # 3)LN层
        self.norm = nn.LayerNorm(d_model)
        # 4)decoder层
        # decoder is shared with embedding layer
        embed_weight = self.embedding.token_embed.weight # Embedding层的weight属性是一个可训练参数
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight

        # 使用nn.Parameter显式为解码器添加偏置项, 作为一个额外的自由参数在模型训练中进行学习
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        """
        :input_ids:   [batch_size, maxlen], 长度为maxlen的token_id序列
        :segment_ids: [batch_size, maxlen], 长度为maxlen的seg_id序列
        :masked_pos:  [batch_size, max_pred], 长度为max_pred的的token_id_pos序列
        :return:
            语言模型LM输出 logits_lm:   [batch_size, max_pred, n_vocab]
            分类CLS输出   logits_clsf: [batch_size, 2]
        """
        # Step1. input_emb
        # output: [batch_size, maxlen, d_model]
        output = self.embedding(input_ids, segment_ids)

        # Step2. EncoderBlock堆叠层
        # output :        [batch_size, maxlen, d_model]
        # enc_self_attn : [batch_size, len_q(=len_k), len_k(=len_q)]
        # it will be decided by first token(CLS)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)

        # Step3.1 语言模型LM的输出 (预测每个被masked的token是什么)
        # masked_pos: [batch_size, max_pred, d_model]
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))

        # get masked position from final output of transformer.
        # BERT专注于序列中的特定位置（被掩盖或标记的词）以便在下游任务中使用这些位置的表示
        # torch.gather: 根据masked_pos索引从 output 中选择对应的元素。结果 h_masked 将包含 output 中由 masked_pos 指定的掩盖位置的表示
        #   第一个参数 output 是要从中提取元素的原始张量
        #   第二个参数 1 指定了要沿哪个维度进行索引提取. 1 表示将沿着序列长度maxlen维度进行索引
        #   第三个参数 masked_pos 是一个索引张量，其形状与 output 的形状相匹配, 除了在索引维度上. 它提供了要提取的特定位置的索引
        # h_masked: [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)
        # h_masked: [batch_size, max_pred, d_model]
        h_masked = self.norm(self.gelu(self.linear(h_masked)))

        # 使用解码器将处理后的掩码位置表示映射到词汇表上，加上偏置项，并输出语言模型的预测。
        # logits_lm: [batch_size, max_pred, n_vocab]
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        # Step3.2 分类CLS的输出
        # 第一个token（[CLS] token）表示进行分类任务, 这个向量用于聚合整个输入序列的信息，并作为分类任务（如情感分析、问答等）的输入
        # h_pooled: [batch_size, d_model]
        h_pooled = self.tanh(self.fc1(output[:, 0, :])) # output[:, 0] == output[:, 0, :]

        # logits_clsf: [batch_size, 2]
        # CLS向量经过一个线性层 self.classifier,以产生最终的分类结果
        logits_clsf = self.fc2(h_pooled)

        return logits_lm, logits_clsf


if __name__ == '__main__':
    # BERT Parameters

    # maximum of length
    maxlen = 30

    # batch size
    batch_size = 6

    # max tokens of prediction
    max_pred = 5

    # number of Encoder Layer
    n_layers = 6

    # number of heads in Multi-Head Attention
    n_heads = 12

    # Embedding Size
    d_model = 768

    # 4*d_model, FeedForward dimension
    d_ff = 768 * 4

    # dimension of K(=Q), V
    d_k = d_v = 64

    n_segments = 2

    text = (
        'Hello, how are you? I am Romeo.\n'
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My baseball team won the competition.\n'
        'Oh Congratulations, Juliet\n'
        'Thanks you Romeo'
    )

    # filter '.', ',', '?', '!'
    # ['hello how are you i am romeo',
    #  'hello romeo my name is juliet nice to meet you',
    #  'nice meet you too how are you today',
    #  'great my baseball team won the competition',
    #  'oh congratulations juliet',
    #  'thanks you romeo'
    #  ]
    sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')

    # 全部训练样本的token
    # ['congratulations', 'i', 'meet', 'too', 'is', 'how', 'name', 'nice', 'thanks', 'today', 'baseball', 'won', 'am', 'the', 'you', 'team', 'great', 'oh', 'my', 'romeo', 'juliet', 'are', 'to', 'hello', 'competition']
    word_list = list(set(" ".join(sentences).split()))

    # 词表(token -> index)
    word_dict = {
        '[PAD]': 0,
        '[CLS]': 1,
        '[SEP]': 2,
        '[MASK]': 3
    }

    # 前4个索引已经被上述特殊标记占用，故从4开始
    for i, w in enumerate(word_list):
        word_dict[w] = i + 4
    print(f"word_dict: {word_dict}")

    # index -> word
    number_dict = {i: w for i, w in enumerate(word_dict)}
    print(f"number_dict: {number_dict}")

    # 词表大小
    vocab_size = len(word_dict)

    # 每个句子对应一个sequence序列
    token_list = list()
    for sentence in sentences:
        # sequence序列
        arr = [word_dict[s] for s in sentence.split()]
        token_list.append(arr)

    # token_list:
    # [[7, 24, 17, 26, 4, 20, 9],
    #  [7, 9, 27, 28, 6, 8, 14, 12, 5, 26],
    #  [14, 5, 26, 22, 24, 17, 26, 25],
    #  [21, 27, 18, 23, 13, 11, 19],
    #  [10, 16, 8],
    #  [15, 26, 9]]
    print(f"token_list: {token_list}")

    # 构造模型
    model = BERT()

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 构造数据
    # input_ids: 长度为maxlen的token_id序列
    # segment_ids: 长度为maxlen的seg_id序列
    # masked_tokens: 长度为max_pred的的token_id序列
    # masked_pos: 长度为max_pred的的token_id_pos序列
    # isNext: 是否下一个segment
    input_ids_mask_batch, batch = make_batch(sentences, token_list, max_pred, number_dict)
    input_ids_bak, input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))
    print(f"segment_ids.shape: {segment_ids.shape}")    # torch.Size([6, 30])
    print(f"input_ids.shape: {input_ids.shape}")        # torch.Size([6, 30])
    print(f"masked_tokens.shape: {masked_tokens.shape}") # torch.Size([6, 5])
    print(f"masked_pos.shape: {masked_pos.shape}")      # torch.Size([6, 5])

    print(f"segment_ids: {segment_ids}")
    # tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    print(f"input_ids_bak: {input_ids_bak}")
    tensor([[1, 25, 14, 7, 5, 17, 28, 19, 2, 6, 26, 12, 2, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 4, 5, 19, 2, 21, 22, 11, 20, 16, 10, 27, 2, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 25, 19, 22, 8, 9, 12, 15, 18, 13, 5, 2, 6, 26, 12, 2, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 6, 26, 12, 2, 4, 5, 19, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 15, 13, 5, 24, 14, 7, 5, 23, 2, 21, 22, 11, 20, 16, 10, 27, 2,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 21, 22, 11, 20, 16, 10, 27, 2, 6, 26, 12, 2, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    print(f"fuck you. input_ids_mask_batch: {input_ids_mask_batch}")
    [['NOTHIT', 'MASK', 'NOTHIT', 'NOTHIT', 'MASK', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD'],
     ['NOTHIT', 'NOTHIT', 'NOTHIT', 'MASK', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'MASK', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD'],
     ['NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'MASK', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'RANDOM', 'NOTHIT', 'NOTHIT', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD'],
     ['NOTHIT', 'MASK', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD'],
     ['NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'MASK', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'MASK', 'NOTHIT', 'MASK', 'NOTHIT', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD'],
     ['NOTHIT', 'MASK', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'NOTHIT', 'MASK', 'NOTHIT', 'NOTHIT', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']]
    print(f"fuck you. len(input_ids_mask_batch): {len(input_ids_mask_batch)}")

    print(f"input_ids: {input_ids}")
    tensor([[1, 3, 14, 7, 3, 17, 28, 19, 2, 6, 26, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 4, 5, 3, 2, 21, 22, 3, 20, 16, 10, 27, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 25, 19, 22, 3, 9, 12, 15, 18, 13, 5, 2, 6, 25, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 3, 26, 12, 2, 4, 5, 19, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 15, 13, 5, 24, 14, 7, 5, 23, 2, 3, 22, 11, 20, 3, 10, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 3, 22, 11, 20, 16, 10, 27, 2, 6, 3, 12, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    print(f"masked_tokens: {masked_tokens}")

    print(f"masked_pos: {masked_pos}")
    masked_tokens: tensor([[25, 5, 0, 0, 0],
                           [19, 11, 0, 0, 0],
                           [8, 26, 0, 0, 0],
                           [6, 0, 0, 0, 0],
                           [21, 27, 16, 0, 0],
                           [26, 21, 0, 0, 0]])
    masked_pos: tensor([[1, 4, 0, 0, 0],
                        [3, 7, 0, 0, 0],
                        [4, 13, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [10, 16, 14, 0, 0],
                        [10, 1, 0, 0, 0]])

    for epoch in range(100):
        # Step1. 梯度清零
        optimizer.zero_grad()

        # Step2. 前向传播
        # 语言模型LM输出 logits_lm:   [batch_size, max_pred, n_vocab]
        # 分类CLS输出   logits_clsf: [batch_size, 2]
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)

        # Step3. 计算损失
        # Step3.1 for masked LM
        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
        print(f"debug. loss_lm.shape: {loss_lm.shape}")
        # debug. loss_lm.shape: torch.Size([])
        loss_lm = (loss_lm.float()).mean()

        # Step3.2 for sentence classification
        loss_clsf = criterion(logits_clsf, isNext)

        # Step3.3 total loss
        loss = loss_lm + loss_clsf

        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        # Step4. 反向传播
        loss.backward()

        # Step5. 更新参数
        optimizer.step()


    # Predict mask tokens ans isNext
    # input_ids: 长度为maxlen的token_id序列
    # segment_ids: 长度为maxlen的seg_id序列
    # masked_tokens: 长度为max_pred的的token_id序列
    # masked_pos: 长度为max_pred的的token_id_pos序列
    # isNext: 是否下一个segment
    input_ids_bak, input_ids_mask, input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(batch[0]))
    print(f"debug. input_ids: {input_ids}")
    # debug. input_ids: tensor(
    #   [[ 1,  3, 18,  3, 21, 25, 10, 23, 17,  6, 13,  2, 19, 27, 10,  2,  0,  0,
    #      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]]
    # )

    print(text)
    print([number_dict[w.item()] for w in input_ids[0] if number_dict[w.item()] != '[PAD]'])
    # ['[CLS]', '[MASK]', 'romeo', '[MASK]', 'name', 'is', 'juliet', 'nice', 'to', 'meet', 'you', '[SEP]', 'oh', 'congratulations', 'juliet', '[SEP]']

    # logits_lm:   [batch_size, max_pred, n_vocab]
    # logits_clsf: [batch_size, 2]
    logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    logits_lm = logits_lm.data.max(2)[1][0].data.numpy() # (5,)
    print(f"debug. after logits_lm.shape: {logits_lm.shape}") # (5,)

    print('masked tokens list : ', [pos.item() for pos in masked_tokens[0] if pos.item() != 0])
    # masked tokens list :  [7, 15]
    print('masked tokens: ', show_words(masked_tokens[0].tolist(), number_dict))
    # masked tokens:  ['won', 'won', 'the', '[PAD]', '[PAD]']
    print('masked tokens: ', show_words([pos.item() for pos in masked_tokens[0] if pos.item() != 0], number_dict))
    # masked tokens:  ['won', 'won', 'the']

    print('predict masked tokens list: ', [pos for pos in logits_lm if pos != 0])
    # predict masked tokens list :  []

    logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
    print('isNext : ', True if isNext else False)
    # isNext :  False

    print('predict isNext : ', True if logits_clsf else False)
    # predict isNext :  False
