1. 如果qk用同一个权重矩阵会怎么样（退化），并行训练原理（masked）
   ```
   在标准的Transformer模型中，查询（Q）、键（K）和值（V）分别通过不同的权重矩阵进行线性变换。这样设计的目的是让模型能够从不同的角度学习输入数据的表示。
   如果查询和键使用同一个权重矩阵，会有以下几个潜在问题：
   表示能力降低：这种设计会限制模型的表示能力。在标准的注意力机制中，通过使用不同的权重矩阵，模型可以从不同的角度学习和匹配查询和键。如果使用同一个权重矩阵，这种多角度的学习能力会受到限制。
   学习难度增加：由于查询和键共享相同的表示，模型可能会更难区分不同的输入，尤其是当输入序列中存在相似或重复的元素时。
   信息损失：在某些情况下，共享权重可能导致信息损失，特别是当查询和键的最优表示本质上是不同的时候
   ```
3. flash attention，kv cache，page attention
   ```
   Flash Attention（闪电注意力）:
   目的：闪电注意力旨在优化大型Transformer模型中注意力机制的计算。它是一种更高效执行注意力计算的方法，尤其是在处理非常长的序列或非常大的模型时。
   工作原理：闪电注意力通常涉及优化注意力分数的计算和存储方式，减少内存占用和计算负担。这可能包括更好的内存管理、并行处理或更有效的算法，以减少注意力计算的复杂性。
   
   KV Cache（键值缓存）:
   目的：在Transformer模型中，特别是在机器翻译或文本生成等应用中，键值缓存用于存储注意力机制中之前计算的键和值对。这在增量解码中特别有用，其中模型一次生成输出的一部分（例如，一个单词或标记）。
   工作原理：在生成新标记时，模型不需要每次都重新计算整个输入序列的注意力，而是重用之前计算的键和值。这种缓存机制显著加快了生成过程，并减少了计算开销。
   
   Page Attention（页面注意力）:
   目的：页面注意力可能用于指代一种组织或批处理注意力计算的方式，特别是在处理大型数据集或长序列的背景下。
   工作原理：这个概念可能涉及将注意力计算划分为更小、更易管理的“页面”或批次。这种划分允许更有效的处理，因为每个“页面”的注意力可以独立或并行计算，减少了内存和计算需求。
   ```

4. 多头attention用途（子空间）
   ```
   捕捉不同子空间的信息：每个“头”可以被看作是独立的注意力机制，它们各自关注输入数据的不同部分或不同方式的关系。这意味着多头注意力可以同时从不同的角度或子空间捕捉输入信息，提高了模型对数据的理解能力。
   
   增加模型的表达能力：通过并行处理多个注意力头，模型可以学习到更加丰富和复杂的特征表示。这对于理解复杂的语言结构尤为重要，如语义上的细微差别、长距离依赖等。
   
   灵活性和泛化能力：多头注意力机制使得模型能够在处理不同类型的任务时更加灵活。比如，在翻译任务中，某些头可能专注于语法结构，而其他头可能专注于词义。这种多方面的关注帮助模型更好地泛化到不同的语言处理任务上。
   
   并行计算提高效率：由于每个头是独立的，它们可以在训练和推理过程中并行计算，这可以显著提高处理速度，尤其是在使用现代GPU和TPU等硬件时。
   
   综上所述，多头注意力机制通过并行处理多个关注点，不仅增强了模型对于不同特征子空间的捕捉能力，也提高了模型的灵活性和计算效率。这使得Transformer模型在处理复杂的序列数据，特别是在自然语言处理领域，表现出色。
   ```
 5. torch写multi head attention  
   ```
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

   ```

2. 对比RNN,LSTM,GRU,transformer3.transformer 中的 cross-attention 和self
   ```
   LSTM旨在处理长期依赖关系时遇到RNN中的梯度消失问题。LSTM通过引入门控机制来控制信息的流动，有效地延长了梯度的传播路径，从而减轻了梯度消失的影响
   GRU是LSTM的简化版，计算效率和内存占用相对改善很多，但是性能差异不大
   transformer引入了自注意力机制，使encoder端后面的列也能看到前面的序列.a-I..高也能够并行计算计算效率有大幅的提升
   ```
3 .transformer 中的 cross-attention 和self-attention 之间的不同attention 之间的不同
   ```
   在encoder-decoder中，self-attention的QKV都是encoder的输入变换来，cross-attention的KV都是来自encoder的输出，Q是decoder的输入经过self-attention后的结果变换而来
   ```

4. transformer的多头和多层的作用
   ```
   多头:每个注意力头可以关注不同的子空间，从而提取不同的特征表示，增加模型对不同特征的敏感度
   多层:通过多个注意力层的级联，实现更深层次的特征提取和组合，这些特征表示经过多次迭代，逐渐提取更抽象、更复杂的特征。多层注意力层的作用是增加模型的深度和非线性能力捕捉更高级别的特征和语义关系，提高模型在复杂任务上的性能
   ```
5. 全连接层为什么要映射至一个更高的维度，又映射回原始维度
   ```
   全连接层映射至一个高维空间来使特征向量在这个空间可分，再映射目标域的原始维度以将学到的“分布式特征表示”映射到样本标记空间
   ```
7. QKV如何得到
   ```
   input经过一个线性层可得到。线性变换的好处: 在QKT部分，线性变换矩阵将KQ投影到了不同的空间，增加了表达能力 (这一原理可以同理SVM中的核函数-将向量映射到高维空间以解决非线性问题)，这样计算得到的注意力矩阵的泛化能力更高
   ```
9. 为什么attention 中的公式是除以根号d_k，不是d k或者其他?
10. transformer 这么多层，是如何解决过拟合问题的?
    ```
    残差连接(同resnet)，解决梯度消失，防止过拟合
    ```
12. 激活函数的作用是?
    ```
    增加模型的非线性表达能力，从而拟合出更复杂的函数
    ```
13. pre-norm和post-norm有什么区别? Bert用的是哪一种?
   ```
   同一设置pre好于post，但单独调其实post可能会更好。pre-norm的优势在于能够给肉E-个绿色通道，不需要对这部分参数正则，防止梯度消失/爆炸，使得训练更容易，而post-norm在残差之后做归一化，对参数正则化的效果更强。bert用的是post-riorm
   ```


在Transformer模型中使用残差连接和Layer Normalization（Layer Norm）而不是Batch Normalization（Batch Norm）有以下原因：

### 1. 残差连接（Residual Connections）

- **梯度传播**：残差连接可以帮助梯度在深层网络中更好地传播。在训练深度网络时，梯度消失和梯度爆炸是常见的问题，残差连接可以通过将输入添加到输出中来保留更多的梯度信息，有助于缓解这些问题。

- **简化训练**：残差连接使得每个层都学习相对于上一层的变化，而不是学习整个映射函数。这样做使得训练更加容易，因为模型只需学习残差（相对于零映射）而不是从头开始学习整个映射。

### 2. Layer Normalization（Layer Norm）

- **适用性**：在深度学习中，Batch Norm通常在卷积神经网络（CNN）中使用较多，而在循环神经网络（RNN）和自注意力模型（如Transformer）中，Layer Norm更为常见。这是因为Batch Norm对于小批量训练数据表现不佳，而Layer Norm更适用于序列数据。

- **数据独立性**：Batch Norm是对小批量数据进行归一化，因此受批量大小的影响，对于序列数据，不同时间步的数据可能具有不同的分布，这会导致Batch Norm的表现不稳定。相比之下，Layer Norm是对每个特征维度进行归一化，因此更适合于不同时间步的序列数据，保持了数据之间的独立性。

- **计算效率**：在Transformer等模型中，由于序列长度可能较长，因此Batch Norm在每个时间步都要对小批量数据进行归一化，计算量较大，而Layer Norm只需要对每个特征维度进行归一化，计算量相对较小，因此更适合于长序列的模型。

综上所述，残差连接和Layer Norm在Transformer中的使用能够更好地促进梯度传播和稳定训练过程，同时也更适用于处理序列数据和长序列模型，提高了模型的训练效率和泛化能力。
