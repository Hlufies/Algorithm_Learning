# Motivation
之前大模型的范式:  
1. Pretrain  
2. Fine-tuning  
```
这种方案，需要对于每个任务都重新 fine-tune 一个新的模型，且不能共用。
但是对于一个预训练的大语言模型来说，这就仿佛好像是对于每个任务都进行了定制化，十分不高效。
是否存在一种方式，可以将预训练语言模型作为电源，不同的任务当作电器，仅需要根据不同的电器（任务），选择不同的插座，对于模型来说，即插入不同的任务特定的参数，就可以使得模型适配该下游任务。
```
**Prompt Learning**就是这个适配器，它能高效得进行预训练语言模型的使用。
```
这种方式大大地提升了预训练模型的使用效率，如下图：
左边是传统的 Model Tuning 的范式：对于不同的任务，都需要将整个预训练语言模型进行精调，每个任务都有自己的一整套参数。
右边是Prompt Tuning，对于不同的任务，仅需要插入不同的prompt 参数，每个任务都单独训练Prompt 参数，不训练预训练语言模型，这样子可以大大缩短训练时间，也极大的提升了模型的使用率。
```
<img width="813" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/6cdf5a50-a9d5-4222-9cfe-fdd813cbf77d">

## Prompt 是什么
```
那么在NLP中 Prompt 代表的是什么呢？

prompt 就是给 预训练语言模型 的一个线索/提示，帮助它可以更好的理解 人类的问题。
例如，下图的BERT/BART/ERNIE 均为预训练语言模型，对于人类提出的问题，以及线索，预训练语言模型可以给出正确的答案。

根据提示，BERT能回答，JDK 是 Oracle 研发的
根据 TL;DR: 的提示，BART知道人类想要问的是文章的摘要
根据提示，ERNIE 知道人类想要问鸟类的能力--飞行
```

```
Prompt 是一种为了更好的使用预训练语言模型的知识，采用在输入段添加额外的文本的技术。

目的：更好挖掘预训练语言模型的能力
手段：在输入端添加文本，即重新定义任务（task reformulation）
```

## Prompt发展路线
```
1. 特征工程阶段
  依赖大量人工
  需要监督数据
2. 架构工程阶段
  人工构建特征的工作量减少
  设计各异的神经网络结构从而拿到结果需要监督数据
3. 预训练-微调阶段
  预训练可以不需要监督数据
```
## 工业界 vs 学术界
```
工业界:更适合工业界的少数据场景，减少一部分标注成本
  工业界对于少数据场景，往往(也有用半监督学习) 先rule-based走通业务逻辑，在线上线下标注数据，再进一步监督学习或微调。
  目前工业界里还难打的过微调
  定位:可以探索
学术界:适合学术研究
  新任务+PL，效果不要太差，都是一次新的尝试与探索
  定位:做研究
```
## Prompt Tuning 工作流程
### 工作流程
  ```
  Prompt 模版（Template）的构造
  Prompt 答案空间映射（Verbalizer）的构造
  文本代入template，并且使用预训练语言模型进行预测
  将预测的结果映射回label。
  ```
## Prompt Tuning 研究方向
### Template 设计研究
  ```
  手工设计
      介绍：人工 手工设计 Template
      优点：直观
      缺点：成本高，需要实验、经验等
  自动学习 模Template板
      介绍：通过模型学习上下文，自动生成 Template
  离散Prompt
      介绍：自动生成自然语言词
      eg: 给定一个大的文本库，给定输入x和输出y，在文本库中离散地搜索出现频繁的中间词或连词等，从而得到一个模板。
  连续Prompt
      介绍：Template的设计不必拘泥于自然语言，直接变成embedding表示也是可以的，设置独立于LM的模板参数，可以根据下游任务进行微调
      eg：给定一个可训练的参数矩阵，将该参数矩阵与输入文本进行连接，从而丢入模型进行训练。
  ```
### Template 形状研究
  ```
  cloze prompt
  介绍：[z] 在句中，适合使用了Mask任务的LM
  prefix prompt
  介绍：[z] 在句末，适合生成LM、自回归LM （自编码LM（Bert） vs 自回归LM（GPT））
  文本匹配任务，Prompt可以有两个[X]
  ```
### Verbalizer 设计研究
  ```
  介绍：寻找合适的答案空间Z，以及答案与标签的映射
  eg：Knowledgeable Prompt-tuning:Incorporating Knowledge intoPrompt Verbalizer for Text Classification (KPT)
  用KB去查询Label相关词作为候选集，然后去噪
  
  注：Label Space Y 是: Positive, Negative, Answer Space Z 可以是表示positive或者negative 的词，
  例如 Interesting/Fantastic/Happy/Boring/1-Star/Bad，具体的 Answer Space Z 的选择范围可以由我们指定。
  可以指定一个 y 对应1-N个字符/词。
  
  具体的答案空间的选择可以有以下三个分类标注：
  
  根据形状
    1.1 Token 类型
    1.2 Span 类型
    1.3 Sentence 类型
  是否有界
    2.1 有界
    2.2 无界
  是否人工选择
    3.1 人工选择
    3.2 自动搜素
      3.2.1 离散空间
      3.2.2 连续空间
  ```
### Pre-trained model select
  ```
  动机：在定义完模版以及答案空间后，需要选择合适的预训练语言模型对 prompt 进行预测，如何选择一个合适的预训练语言模型也是需要人工经验判别的。
  具体的预训练语言模型分类：
  autoregressive-models: 自回归模型，主要代表有 GPT，主要用于生成任务；
  autoencoding-models: 自编码模型，主要代表有 BERT，主要用于NLU任务；
  seq-to-seq-models：序列到序列任务，包含了an encoder 和 a decoder，主要代表有 BART，主要用于基于条件的生成任务，例如翻译，summary等；
  multimodal-models：多模态模型
  retrieval-based-models：基于召回的模型，主要用于开放域问答

  ```
###  Expanding the Paradigm（范式拓展）
**如何对已有的 Prompt 进行任务增强以及拓展**
  ```
  1. Prompt Ensemble：Prompt 集成，采用多种方式询问同一个问题
  2. Prompt Augmentation：Prompt 增强，采用类似的 prompt 提示进行增强
  3. Prompt Composition：Prompt 组合，例如将一个任务，拆成多个任务的组合，比如判别两个实体之间是否是父子关系，首先对于每个实体，先用Prompt 判别是人物，再进行实体关系的预测。
  4. Prompt Decomposition：将一个prompt 拆分成多个prompt
  ```
### 训练策略
  ```
  动机：Prompt-based 模型在训练中，有多种训练策略，可以选择哪些模型部分训练，哪些不训练
根据训练数据区分：
  Zero-shot: 对于下游任务，没有任何训练数据
  Few-shot: 对于下游任务只有很少的训练数据，例如100条
  Full-data: 有很多的训练数据，例如1万多条数据
根据不同的参数更新的部分：
  预训练模型
  Prompts 参数
    预训练语言模型：可以选择精调，或者不训练
对于prompts：
  Tuning-free Prompting
    直接做zero-shot
  Fixed-LM Prompt Tuning
    引入额外与Prompt相关的参数，固定LM参数，微调与Prompt相关参数
  Fixed-prompt LM Tuning
    引入额外与Prompt相关的参数，固定与Prompt相关参数，微调LM
  Prompt + LM Tuning
  引入额外与Prompt相关的参数，两者都微调
策略选择
  数据量级是多少
  是否有个超大的 Left-to-right 的语言模型
（注：通常如果只有很少的数据的时候，往往希望不要去 fine-tune 预训练语言模型，而是使用LM的超强能力，只是去调prompt 参数。而让数据量足够多的时候，可以精调语言模型。）
  ```
