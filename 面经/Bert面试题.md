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

## Bert预训练

BERT (Bidirectional Encoder Representations from Transformers) 是一种双向 Transformer 语言模型，其预训练方法包括两个主要任务：Masked Language Model (MLM) 和 Next Sentence Prediction (NSP)。这两个任务共同帮助 BERT 理解语言的上下文和句子关系。以下是 BERT 的详细预训练方法：

### 1. Masked Language Model (MLM)

MLM 任务旨在通过掩码（masking）部分输入序列中的词，训练模型预测这些被掩码的词。具体步骤如下：

#### 步骤：
1. **输入序列处理**：
   - 随机选择 15% 的词进行掩码处理。
   - 对于被选择的词，有 80% 的概率替换为 `[MASK]`，10% 的概率替换为随机词，10% 的概率保持原词。

2. **模型训练**：
   - 使用掩码后的输入序列，通过 Transformer 编码器生成上下文表示。
   - 训练模型预测被掩码词的真实词。

#### 例子：
输入序列："The quick brown fox jumps over the lazy dog"
掩码处理后："The quick [MASK] fox jumps over [MASK] lazy dog"

模型训练目标是预测 `[MASK]` 的真实词分别是 "brown" 和 "the"。

### 2. Next Sentence Prediction (NSP)

NSP 任务旨在训练模型理解句子之间的关系。具体步骤如下：

#### 步骤：
1. **输入句子对**：
   - 从语料库中选择句子对 (A, B)。
   - 50% 的句子对 (A, B) 实际是连续的句子，标记为 "IsNext"。
   - 50% 的句子对 (A, B) 是不连续的随机句子，标记为 "NotNext"。

2. **模型训练**：
   - 将句子对 (A, B) 拼接成一个输入序列，并在序列开头添加 `[CLS]` 标记，句子 A 和 B 之间添加 `[SEP]` 标记。
   - 使用 BERT 模型生成整个序列的表示。
   - 使用 `[CLS]` 标记的输出表示进行二分类任务，预测句子对是否为连续句子。

#### 例子：
输入句子对 (A, B):
A: "The quick brown fox jumps over the lazy dog"
B: "The dog is not amused"

如果 (A, B) 实际是连续句子，则标记为 "IsNext"；如果是随机组合，则标记为 "NotNext"。

### 3. 预训练过程

BERT 的预训练过程结合了 MLM 和 NSP 两个任务。具体步骤如下：

1. **数据准备**：
   - 使用大量的无监督文本语料库（如维基百科、图书语料库等）。
   - 对文本进行标记化处理，将每个单词或子词转换为 ID 表示。

2. **模型架构**：
   - 使用多层双向 Transformer 编码器，每层包含自注意力机制和前馈神经网络。
   - 在模型顶部添加特定任务层：MLM 任务层和 NSP 任务层。

3. **损失函数**：
   - MLM 任务和 NSP 任务的损失函数之和作为模型的总损失。
   - 通过梯度下降算法（如 Adam 优化器）最小化总损失。

#### 例子：
```python
from transformers import BertTokenizer, BertForPreTraining, AdamW
import torch

# 初始化 tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')

# 准备输入数据
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors='pt')

# Masked Language Model (MLM) 处理
inputs['labels'] = inputs.input_ids.detach().clone()
random_indexes = torch.rand(inputs.input_ids.shape).uniform_(0, 1) < 0.15
inputs.input_ids[random_indexes] = tokenizer.mask_token_id

# Next Sentence Prediction (NSP) 处理
next_sentence = "The dog is not amused."
next_inputs = tokenizer(next_sentence, return_tensors='pt')
inputs['next_sentence_label'] = torch.tensor([1])  # 假设为连续句子

# 训练模型
outputs = model(**inputs, next_sentence_label=inputs['next_sentence_label'])
loss = outputs.loss

# 优化器和梯度更新
optimizer = AdamW(model.parameters(), lr=5e-5)
loss.backward()
optimizer.step()
```

### 4. 预训练效果

通过上述预训练方法，BERT 能够学习丰富的语言表示，在多种下游任务（如文本分类、问答系统、命名实体识别等）中表现出色。预训练后的 BERT 模型可以通过微调（Fine-tuning）适应特定任务，从而进一步提升性能。

### 总结

BERT 的预训练方法结合了 Masked Language Model (MLM) 和 Next Sentence Prediction (NSP) 两个任务，通过在大规模无监督语料库上的训练，学习到语言的深层次表示。使用这些预训练表示，可以显著提升下游任务的性能。

虽然MLM（Masked Language Model）和NSP（Next Sentence Prediction）是BERT预训练的两个关键任务，但它们也各自存在一些缺点：

### 缺点

#### 1. Masked Language Model (MLM) 的缺点：

- **信息丢失**：在训练过程中，由于部分词被掩码，模型可能无法获取完整的上下文信息，从而导致信息丢失。
- **预测偏差**：掩码词的预测可能受到附近词的影响，导致预测结果偏差，尤其是在上下文模糊或含糊的情况下。

#### 2. Next Sentence Prediction (NSP) 的缺点：

- **任务偏向**：NSP任务只关注句子之间的关系，可能忽略了单个句子内部的语义和逻辑，导致模型在某些任务上表现不佳。
- **二分类问题**：NSP任务是一个二分类问题，只考虑了两个句子是否连续，而忽略了句子之间的更复杂的关联性。

### 克服方法

虽然MLM和NSP存在一些缺点，但可以通过以下方式来克服或减轻这些问题：

#### 1. 多任务预训练：

结合MLM和NSP以及其他自监督任务，形成多任务预训练的框架，从而综合利用不同任务的优势，提高模型的泛化能力和性能。

#### 2. 对抗式预训练：

通过引入对抗性训练，使模型在预训练阶段学习到更具鲁棒性的表示，从而提高模型对干扰的抵抗能力，减少信息丢失的影响。

#### 3. 自监督任务设计：

设计更加有效的自监督任务，以提高模型在不同层次上的表示学习能力，从而更好地捕捉文本的语义和逻辑信息。

#### 4. 数据增强和样本策略：

通过数据增强和样本策略，扩大训练数据的覆盖范围，增加模型在不同场景下的泛化能力，从而减轻任务偏向性的影响。


https://cloud.tencent.com/developer/article/1588531

#### 5. 结合有监督学习：

在预训练之后，通过有监督学习的方式进一步微调模型，使其适应具体的下游任务要求，从而更好地解决特定任务的问题。

综上所述，虽然MLM和NSP存在一些缺点，但可以通过合理的方法和策略来克服这些问题，并进一步提升预训练模型的性能和泛化能力。
