import torch
import torch.nn as nn
import numpy as np
from utils import PositionalEncoding
from utils import get_attn_pad_mask
from utils import get_attn_subsequence_mask
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionWiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            # d_model = 512 => input and output's dim
            # d_ff
            nn.Linear(in_features=d_model, out_features=d_ff, bias=False),
            nn.ReLU(),
            # 意味着这个线性层将只进行权重矩阵和输入的乘法操作，而不加任何常数偏置。
            # 这样做的理由可能是出于上述的某一点或者特定于模型设计的其他考虑。bias=False
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model
        self.d_ff = d_ff
    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        # 这里为什么要用LayerNorm
        # fc的ouput+redisudal之后,再过一层 LayerNorm
        return nn.LayerNorm(self.d_model)(output + residual) # [batchSize, seq_len, d_model]
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: to match others                [batch, n_heads, len_q, d_q]
        :param K: to be matched                  [batch, n_heads, len_k, d_k]
        :param V: information to be extracted    [batch, n_heads, len_v, d_v]
        :param attn_mask:
        :return:
        """

        '''
        Q.shape                   : torch.Size([batchSize, n_heads, len_q, d_q])
        K.transpose(-1, -2) shape : torch.Size([batchSize, n_heads, len_k, d_k])
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores :   [batchSize, n_heads, len_q, len_k]

        # Fills elements of self tensor with value where mask is True.     # attn_mask: [batchSize, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores) # attn:    [batchSize, n_heads, len_q, len_k]
        context = torch.matmul(attn, V)   # context: [batchSize, n_heads, len_q, d_v]
        return context, attn
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc =  nn.Linear(n_heads * d_v, d_model, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention(d_k=d_k)
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.d_model = d_model
    def forward(self, input_Q, input_K, input_V, attn_mask):
        """

        :param input_Q:  [batch_size, len, d_model]
        :param input_K:  [batch_size, len, d_model]
        :param input_V:  [batch_size, len, d_model]
        :param attn_mask:[batch_size, len_1, len_2]
        :return: enc_outputs [batchSize, len, d_model], enc_self_attention [batchSize, len, len1, len2]
        """

        residual, batch_size = input_Q, input_Q.size(0)

        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        '''
        transpose 方法用于交换张量的两个维度。
        这里将第二个维度（seq_len）和第三个维度（n_heads）交换。
        交换后的张量形状变为 [batch_size, n_heads, seq_len, d_k]。
        这样做的目的是为了将不同的头放到一个单独的维度上，以便后续的操作能够并行处理每个头
        '''

        '''
        view 方法
        用途：view 用于改变张量的形状而不改变其数据。
        工作方式：当你使用 view 方法时，它返回一个新的张量，这个新张量与原始张量共享数据但具有不同的形状。它不会改变张量中元素的顺序或位置。
        限制：要使用 view，原始张量必须是连续的（内存中的元素排列没有间断）。如果不是，可能需要先调用 .contiguous()。
        例子：假设有一个形状为 [4, 3] 的张量（即 4 行 3 列），你可以使用 view 将其重塑为 [3, 4]（3 行 4 列），但元素的顺序保持不变。
        transpose 方法
        用途：transpose 用于交换张量中的两个维度。
        工作方式：它不是简单地重塑张量，而是交换指定的两个维度。这意味着张量中元素的顺序会发生改变。
        应用：transpose 常用于需要改变数据布局的场景，例如，在处理图像数据时从 HWC（高度、宽度、通道）格式转换为 CHW（通道、高度、宽度）格式。
        例子：如果有一个形状为 [4, 3] 的张量，使用 transpose 交换这两个维度后，形状将变为 [3, 4]，但与 view 不同，这里元素的实际顺序发生了改变
        '''
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batchSize, len_q, n_heads*d_q] -> [batchSize, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batchSize, len_k, n_heads*d_k] -> [batchSize, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batchSize, len_v, n_heads*d_v] -> [batchSize, n_heads, len_v(=len_k), d_v]

        '''
        unsqueeze(1):
        这个方法在 attn_mask 张量的第二个维度（索引为 1）处添加一个新的维度。
        假设原始的 attn_mask 的形状是 [batchSize, seq_len, seq_len]，那么 unsqueeze(1) 之后，其形状变为 [batchSize, 1, seq_len, seq_len]。
        这一步是为了让掩码张量的维度与多头注意力机制中期望的维度对齐。在多头注意力机制中，每个头对应于不同的维度，因此需要在掩码张量中添加一个额外的维度来表示这些头。
        repeat(1, self.n_heads, 1, 1):
    
        repeat 方法用于沿指定的维度重复张量的元素。
        在这个例子中，1, self.n_heads, 1, 1 表示在第一个维度（batchSize）重复一次，在第二个维度（新增的维度，表示头的数量）重复 self.n_heads 次，在第三和第四个维度（seq_len）重复一次。
        这样做的结果是，原始的掩码张量被复制了 self.n_heads 次，形成一个新的张量，其形状变为 [batchSize, n_heads, seq_len, seq_len]
        '''
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # attn_mask : [batchSize, n_heads, seq_len, seq_len]
        context, attn = self.ScaledDotProductAttention(Q, K, V, attn_mask)# context:    [batchSize, n_heads, len_q, d_v], attn: [batchSize, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batchSize, len_q, n_heads * d_v]
        output = self.fc(context) # [batchSize, len_q, d_model]
        return nn.LayerNorm(self.d_model)(output + residual), attn # enc_outputs [batchSize, len, d_model], enc_self_attention [batchSize, len, len1, len2]
"""
#====================================================#
#Eecoder#
#====================================================#
"""
class EncoderLayer(nn.Module):
    def __init__(self,d_model, d_k, d_v, n_heads,d_ff):
        super(EncoderLayer, self).__init__()
        self.encoder_self_attention = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)
        self.pos_ffn = PositionWiseFeedForwardNet(d_model=d_model, d_ff=d_ff)
    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs        : [batchSize, src_len, d_model]
        :param enc_self_attn_mask: [batchSize, src_len, src_len]
        :return:
        """
        enc_outputs, attn = self.encoder_self_attention(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V => enc_outputs: [batchSize, ]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_layers, d_k, d_v, n_heads, d_ff):
        super(Encoder,self).__init__()
        # nn.Embedding(num_embeddings,embedding_dim,)
        self.source_embedding = nn.Embedding(src_vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model=d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, n_heads,d_ff) for _ in range(n_layers)])

    def forward(self,encoder_inputs):
        """
        :param encoder_inputs: [batchSize, src_len]
        :return encoder_outputs, encoder_self_attentions: [batchSize,src_len,d_model], [batchSize]
        """
        word_embedding = self.source_embedding(encoder_inputs)                                        # word_embeding     : [batchSize, src_len, d_model]
        position_embedding = self.position_embedding(word_embedding.transpose(0, 1)).transpose(0, 1)  # position_embedding: [batchSize, src_len, d_model]
        encoder_outputs = word_embedding + position_embedding
        # encoder_outputs   : [batchSize, src_len, d_model]
        encoder_self_attention_mask = get_attn_pad_mask(encoder_inputs,encoder_inputs)                #                   : [batchSize, src_len, src_len]
        print('encoder_self_attention_mask',encoder_self_attention_mask)
        encoder_self_attentions = []
        for layer in self.layers:
            encoder_outputs, encoder_self_attention = layer(encoder_outputs, encoder_self_attention_mask) # [batchSize, len_q, d_model], [batchSize, n_heads, len_q, len_k]
            encoder_self_attentions.append(encoder_self_attention)
        return encoder_outputs, encoder_self_attentions

"""
#====================================================#
#Decoder#
#====================================================#
"""

class DecoderLayer(nn.Module):
    def __init__(self,d_model, d_k, d_v, n_heads,d_ff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.dec_enc_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PositionWiseFeedForwardNet(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # dec_outputs: [batchSize, tgt_len, d_model], dec_self_attn: [batchSize, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batchSize, tgt_len, d_model], dec_enc_attn: [batchSize, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batchSize, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn
class Decoder(nn.Module):
    def __init__(self,tgt_vocab_size,d_model,n_layers,d_k, d_v, n_heads,d_ff):
        super(Decoder,self).__init__()
        self.target_embedding = nn.Embedding(tgt_vocab_size,d_model)
        self.position_embedding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_k, d_v, n_heads,d_ff) for _ in range(n_layers)])
    def forward(self,decoder_inputs,encoder_inputs,encoder_outputs):
        word_embedding = self.target_embedding(decoder_inputs).to(device)  # [batchSize, tgt_len, d_model]
        # print(f'word_embedding transpose(0,1) shape {word_embedding.transpose(0, 1).shape}')

        position_embedding = self.position_embedding(word_embedding.transpose(0, 1)).transpose(0, 1).to(device)
        decoder_outputs = word_embedding + position_embedding

        '''
        decoder_self_attention_pad_mask = get_attn_pad_mask(decoder_inputs, decoder_inputs).to(device)
        这行代码生成一个自注意力掩码，用于遮盖解码器输入中的填充（PAD）位置。
        掩码是通过 get_attn_pad_mask 函数生成的，这个函数基于输入序列创建一个掩码，使得模型在计算注意力时忽略填充位置
        '''
        decoder_self_attention_pad_mask = get_attn_pad_mask(decoder_inputs,decoder_inputs).to(device)

        '''
        decoder_self_attention_subsequence_mask = get_attn_subsequence_mask(decoder_inputs).to(device)
        生成一个子序列掩码，用于遮盖解码器中未来的位置，以防止信息泄露。
        '''
        decoder_self_attention_subsequence_mask = get_attn_subsequence_mask(decoder_inputs).to(device)
        # print(f'decoder_self_attention_subsequence shape {decoder_self_attention_subsequence_mask}')

        '''
        decoder_self_attention_mask = torch.gt(decoder_self_attention_pad_mask + decoder_self_attention_subsequence_mask, 0)
        这行代码结合了填充掩码和子序列掩码，创建了一个综合的自注意力掩码。
        使用 torch.gt 生成一个布尔掩码，表示哪些位置应被忽略。
        '''
        decoder_self_attention_mask = torch.gt(decoder_self_attention_pad_mask+decoder_self_attention_subsequence_mask,0)
        print(f'decoder self attention mask {decoder_self_attention_mask}')

        '''
        这行代码为解码器-编码器的注意力机制生成掩码，用于遮盖编码器输入中的填充位置。
        这对于处理长度不一的源序列和目标序列非常重要
        '''
        decoder_encoder_attention_mask = get_attn_pad_mask(decoder_inputs,encoder_inputs)
        print(f'decoder_encoder_attention_mask shape: {decoder_encoder_attention_mask}')

        decoder_self_attentions = []
        decoder_encoder_attentions = []
        for layer in self.layers:
            decoder_outputs, decoder_self_attention,decoder_encoder_attention = layer(decoder_outputs,encoder_outputs,decoder_self_attention_mask,decoder_encoder_attention_mask)
            decoder_self_attentions.append(decoder_self_attention)
            decoder_encoder_attentions.append(decoder_encoder_attention)
        return decoder_outputs, decoder_self_attentions, decoder_encoder_attentions
class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,n_layers, n_heads,d_model,d_k, d_v, d_ff):
        super(Transformer,self).__init__()
        """
        :param src_vocab_size : 原词汇的大小, 输入文本的大小
        :param tgt_vocab_size : 目标词汇的大小
        :param n_layers       : 参数指定了 Transformer 模型中编码器和解码器的层数。每层通常包括自注意力机制和前馈神经网络
        :param nn_heads       : 头的数量（在多头注意力机制中）。这个参数决定了每层中多头自注意力部分的头数。每个头学习输入数据的不同方面
        :param d_model        : 这是 Transformer 模型中所有层的输入和输出维度。它也是内部层（如自注意力层）的输入和输出维度
        :param d_k            : 键（Key）/查询（Query）维度。在多头注意力机制中，输入数据被线性映射到一组键、值和查询上。d_k 指的是键和查询的维度。
        :param d_v            : 值（Value）维度。同样在多头注意力机制中，d_v 指的是值的维度
        :param d_ff           : 前馈神经网络的维度。这是 Transformer 中每个编码器和解码器层里面前馈神经网络（Feed-Forward Network, FFN）的内部维度
        """
        self.Encoder = Encoder(src_vocab_size=src_vocab_size,n_heads=n_heads, n_layers=n_layers, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff,).to(device)
        self.Decoder = Decoder(tgt_vocab_size=tgt_vocab_size,n_heads=n_heads, n_layers=n_layers, d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff,).to(device)
        self.projection = nn.Linear(in_features=d_model, out_features=tgt_vocab_size,bias=False).to(device)
    def forward(self,encoder_inputs,decoder_inputs):

        """
        :param encoder_inputs: [batchSize, src_len]
        :param decoder_inputs: [batchSize, tgt_len]
        :return:
        """

        '''
         enc_inputs作为输入 形状为[batch_size, src_len]，
         输出由自己的函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，可以是特定每一层的输出；也可以是中间某些参数的输出；
         enc_outputs就是主要的输出，enc_self_attns这里没记错的是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词相关性；
        '''
        encoder_outputs, encoder_self_attentions = self.Encoder(encoder_inputs)

        '''
        dec_outputs 是decoder主要输出，用于后续的linear映射；
        dec_self_attns类比于enc_self_attns 是查看每个单词对decoder中输入的其余单词的相关性；
        dec_enc_attns是decoder中每个单词对encoder中每个单词的相关性；
        '''
        # decoder_inputs,
        decoder_outputs, decoder_self_attentions, decoder_encoder_attentions = self.Decoder(decoder_inputs,encoder_inputs,encoder_outputs)

        decoder_logits = self.projection(decoder_outputs)

        return decoder_logits.view(-1,decoder_logits.size(-1)), encoder_self_attentions, decoder_self_attentions, decoder_encoder_attentions



        
    



