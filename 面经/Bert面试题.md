## Bert的结构是什么?一般可以做什么任务?
1. BERT的核心模型架构由Transformer的编码器 (encoder) 层堆叠而成，它采用的是**全双向 (bi-directional) 结构**。
2. 分类/回归任务: 利用BERT模型输出的“pooler output”，即特殊的[CLS]标记 (放置在每个输入序列前的第一个位置)的最终输出隐藏状态，再经过一个额外的全连接层和一个tanh激活函数。在下游任务的微调过程中，[CLS]标记的隐藏状态经过一个线性投影层，然后根据具体任务通过softmax (用于多类别分类)或sigmoid(用于二分类或回归任务) 层生成最终预测。
3. 序列级任务: 对于如命名实体识别 (NER)或问答 (QA) 等需要针对输入序列中每个token给出预测的任务，利用BERT输出的“sequence output”，即每层Transformer编码器对每个token生成的隐藏状态。这些隐藏状态可以直二一一续的任务特定层，如条件随机场 (CRE) 层或简单的全连接层，以产生每个token层面的预测。

## Bert与传统的文本表示模型 (如word2vec或GloVe) 有什么不同?简单来说，Bert是动态的而word2vec是静态的
1. Word2Vec通过训练为每个词汇提供了一个固定的词向量表示，这些词向量一经训练便不再变化。
2. Bert根据输入句子中的具体上下文动态计算每个词的嵌入
## Bert和GPT有什么区别?
1. 结构上:bert 只使用了Transformer的encoder部分，gpt仅使用了Transformer的decoder部分，但去掉了masked self-attention层之后的层
2. 训练任务上:bert主要使用mask任务训练，类似于“填空题”;gpt主要做的任冬是next token prediction任务。前者是双向，同时考虑左侧和右侧的上下人三后者是单向，只能使用之前的token作为上下文

## 它的训练任务有什么?仔细说一下
1. MLM，随机MASK 15%的tokens,
(1)80%的概率替换成[MASK]
(2)10%的概率是替换为随机的token
(3)10%的概率是保持不变
2. .NSP，在NSP任务中，训练样本由一对句子组成，这对句子有50%的概率实际上是相邻的句子，有50%的概率第二个句子是随机选择的，与第一个句子不是连续的
## Bert的mask方法有什么缺陷吗?
bert的mask方法是token级的，但因为其对词的切分方法是BPE，即token可能是个单词的子部分，那么对于一个单词只mask子词部分，那么就非常容易预测了所以后面有几篇文章提出了span mask(GLM)，即将这个子词所在的单词整一个mask。用到中文上时，百度还提出了命名实体mask。
## Bert的NSP是怎么预测的?
取pooling层后的[CLS] token并经过一个投影层(batch_size，dim)->(batch_size,2)，再做一个交叉损失计算loss，与mlm任务的loss加一起优化
## Bert的参数量计算
Bert参数量计算，简要说一下，不用直接得出计算数字，给每层的公式就行
```
a.token embedding参数矩阵: vocab_length词表长度*dim特征维度;
  position embedding参数矩阵: 512*dim特征维度;
  segment embedding参数矩阵: 2*dim特征维度
```
```
b.WQkv+W。多头注意力头输出concat后的加权投影层+各bias参数矩阵:[4dim^2+4dim]*layers_nums层数，base是12层
```
```
c.Layernorm:embedding后一个，FFN后一个，att后一个，共2dim+[2*2dim]*layers_nums
```
```
d.FFN: 2个linear和其bias，dim->4dim->dim，共[8dim^2+5dim]*layers_nums
e.pooling层: dim^2+dim
f.MLM、NSP涉及的参数:分别是768 * vocab_size+ 768 *2
```
## Bert的layernorm是BN还是LN? 是pre-norm还是post-norm?
LN，post-norm  
[这里细谈LayerNorm和BatchNorm的细节点]()  
[Pre-Norm和Post-Norm的细节点]()  

## Bert的embedding部分和原transformer的有什么不同?
a. Bert的embedding部分是由三部分组成的，token embedding+positionembedding+segment embedding ;
b.原transformer缺少segment embedding，而且position embedding上使用的是固定式的三角函数计算出的，bert的是学习式的，即学习了一个position的参数矩阵。
## Bert的三部分embedding为什么是相加而不是concat?
1. concat导致模型参数的显著增加 (如自注意力层的权重矩阵Wa/K/v) : 加操作确保了嵌入向量的维度保持不变。如果将这些嵌入进行拼接，那么最终的嵌入维度将是各部分嵌入维度之和，以至于后面需要调整模型的后续部分参数矩阵维度导致参数增加
2. concat导致模型需要学习不同纬度上的不同特征: 相加操作允许模型在每个位置上融合来自不同嵌入空间的信息，而这些信息是同时考虑的。这种方·以励艺型学习如何整合词语的含义、其在句子中的位置以及它属于哪个句子段的后忌。
3. concat实验验证收益不大
