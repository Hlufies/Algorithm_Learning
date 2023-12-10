import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_model, tgt_vocab_size):
        '''
        在 Python 中，super(Transformer, self).__init__() 这行代码是用于调用父类（或超类）的构造函数的一种方式。这种语法特别常见于继承结构中。
        具体到这个例子，假设 Transformer 是一个继承自另一个类（比如 SomeParentClass）的子类，这行代码的作用如下：
        调用父类的构造函数：super(Transformer, self) 返回一个代表父类的对象，而 .__init__() 则是调用这个父类对象的构造函数。这是初始化父类的一部分或全部状态的标准方式。
        确保父类被正确初始化：在多重继承的情况下，确保所有父类都被正确初始化是很重要的。使用 super() 可以避免在继承链中多次调用父类的构造函数，这种情况在不使用 super() 时可能发生。
        维护继承链：super() 的使用有助于维护和管理继承体系中的方法解析顺序（Method Resolution Order, MRO）。它按照一定的顺序调用父类，这对于复杂的继承结构尤其重要。
        '''
        super(Transformer, self).__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder()

        # projection: 输出层 d_model 是我们解码层每个token输出的维度大小，之后会做一个 tgt_vocab_size 大小的softmax
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        # enc_inputs: [batchsize, src_len]
        # dec_inputs: [batchsize, tgt_len]

        '''
        enc_inputs作为输入 形状为[batch_size, src_len]，
        输出由自己的函数内部指定，想要什么指定输出什么，可以是全部tokens的输出，可以是特定每一层的输出；也可以是中间某些参数的输出；
        enc_outputs就是主要的输出，enc_self_attns这里没记错的是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词相关性；
        '''
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)

        # dec_outputs 是decoder主要输出，用于后续的linear映射；
        # dec_self_attns类比于enc_self_attns 是查看每个单词对decoder中输入的其余单词的相关性；
        # dec_enc_attns是decoder中每个单词对encoder中每个单词的相关性；
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_outputs做映射到词表大小
        # dec_logits: [batch_size, src_vocab_size, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)









class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()