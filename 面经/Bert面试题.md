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

![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/da9e35c1-3549-4d64-a505-d7c0fde69ea7)  
### Token Embeddings
token embedding 层是要将各个词转换成固定维度的向量。在BERT中，每个词会被转换成768维的向量表示。  
### Segment Embeddings
BERT 能够处理对输入句子对的分类任务。这类任务就像判断两个文本是否是语义相似的。句子对中的两个句子被简单的拼接在一起后送入到模型中。那BERT如何去区分一个句子对中的两个句子呢？答案就是segment embeddings.  
### Position Embeddings
BERT包含这一串Transformers (Vaswani et al. 2017)，而且一般认为，Transformers无法编码输入的序列的顺序性。总的来说，加入position embeddings会让BERT理解下面下面这种情况：
```
I think, therefore I am
第一个 “I” 和第二个 “I”应该有着不同的向量表示。

BERT能够处理最长512个token的输入序列。论文作者通过让BERT在各个位置上学习一个向量表示来讲序列顺序的信息编码进来。

Position Embeddings layer 实际上就是一个大小为 (512, 768) 的lookup表，表的第一行是代表第一个序列的第一个位置，第二行代表序列的第二个位置，以此类推。因此，如果有这样两个句子“Hello world” 和“Hi there”, “Hello” 和“Hi”会由完全相同的position embeddings，因为他们都是句子的第一个词。同理，“world” 和“there”也会有相同的position embedding。
```
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
## 为什么bert三个embedding可以相加
```
这虽然在深度神经网络里变得非常复杂，本质上神经网络中每个神经元收到的信号也是“权重”相加得来。具体细节的分析这里就不提了，有兴趣的同学可以自己推一推。
这里想说一下宽泛一点的分析(瞎扯)。
在实际场景中，叠加是一个更为常态的操作。比如声音、图像等信号。一个时序的波可以用多个不司频率的正弦波叠加来表示。只要叠加的波的频率不同，我们就可以通过傅里叶变换进行逆向转换。
一串文本也可以看作是一些时序信号，也可以有很多信号进行叠加，只要频率不同，都可以在后面的复杂神经网络中得到解耦(但也不一定真的要得到解耦)。在BERT这个设定中，token,segment，position明显可以对应三种非常不同的频率
由此可以再深入想一想，在一串文本中，如果每个词的特征都可以用叠加波来表示，整个序列又可以进一步叠加。哪些是低频信号 (比如词性?)，哪些是高频信号 (比如语义? )，这些都隐藏在embedding中，也可能已经解耦在不同维度中了。说不定可以是一种新的表示理论:)
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
