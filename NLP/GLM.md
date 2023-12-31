# 一. Conlusion（Framework & advantage & disadvantage）
1. GLM -> **Framework: autoregressive blank infilling** -> **advantage**: TO solve the problem of NLU AND NLG 
2. autoregressive: GPT: **Framework**:left->right language model -> **advantage**: long-text generation and show fewshot learning ability -> **disadvantage**: is the unidirectional attention mechanism, which **cannot fully capture the dependencies between the context words in NLU tasks**.
3. autoencoding: BERT -> **Framework**： learn bidirectional context encoders via denoising objectives, e.g.**Masked Language Model (MLM)** -> **advantage**: suit natural language understanding tasks -> **disadvantage**: could not be directly applied for text generation(NLG).
4. encoder-decoder: T5 -> ： **Framework**adopt bidirectional attention for the encoder, unidirectional attention for the decoder, and cross attention between them -> conditional generation tasks -> **advantage**: text summarization and response generation —> **disadvantage**: 但T5为了达到和RoBERTa和DeBERTa相似的性能，往往需要更多的参数量。
5. 三者的训练目标

   ```
   GPT的训练目标是从左到右的文本生成。
   BERT的训练目标是对文本进行随机掩码，然后预测被掩码的词。
   T5则是接受一段文本，从左到右的生成另一段文本。  
   ```
  
# 二. Improvement 
1. span shuffling
2. 2D positional encoding.
# 三. Pretraining Framework
我们提出了一种基于新颖的自回归空白填充目标的通用预训练框架 GLM。 GLM 将 NLU 任务制定为包含任务描述的完形填空问题，可以通过自回归生成来回答。
## 3.1 Pretraining Objective
这篇文章提出一个自回归空格填充的任务（Autoregressive Blank Infifilling），来兼容三种预训练目标。自回归填充有些类似掩码语言模型，首先采样输入文本中部分片段，将其替换为[MASK]标记，然后预测[MASK]所对应的文本片段。与掩码语言模型不同的是，预测的过程是采用自回归的方式  
### 3.1.1 Autoregressive Blank Infilling  
1. 当被掩码的片段长度为1的时候，空格填充任务等价于掩码语言建模；  
2. 当将文本1和文本2拼接在一起，然后将文本2整体掩码掉，空格填充任务就等价于条件语言生成任务。  
3. 当全部的文本都被掩码时，空格填充任务就等价于无条件语言生成任务
```
原文为[x1,x2,x3,x4,x5,x6]，对两个跨度进行采样B = [x3], [x5,x6]。
（从参数为3的柏松分布中随机抽取长度跨度，直到15%的tokens被mask掉）
（为什么：经验发现15%对下游NLU任务性能很重要）
将采样部分替换为[MASK].  --->A = [x1,x2,[MASK], x4,[MASK]],并对B进行shuffle排列
GLM自回归生成B部分，每个span输入以[S] 作为开头 [E]作为输出结尾，并使用2D位置编码
position 1 表示跨度间的位置
position 2 表示跨度内的位置
自回归生成过程中，A可以实现自我的全注意力，但是不能关注B，B可以关注A以及在B中前面的spans。
```
```
为什么？
我的理解是，设计一种包含双向和单向注意力的架构
```
整体架构是基于 Transformer，主要进行 3 处修改，分别为：  
1. 调整了 norm 以及 残差连接 的顺序：这个主要是 Post-Norm vs. Pre-Norm 的问题，在大模型的各类应用当中，Post-Norm 的作法比较常见（例如 Bert），具体分析可以见下，Pre-Norm 从某种程度上加深了网络的宽度而降低了深度，导致效果变差。
2. 使用了单个的 linear layer 预测 token 输出。
3. 使用了 GeLU 替换了 ReLU：GeLU 在负数部分也有激活值，一定程度上解决神经元死亡问题，正数部分和Relu一样
   
<img width="574" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/fb8a8cdd-c712-4aa5-8bbc-40ecbba2a62a">

### 3.1.2 Multi-Task Pretraining
作者使用了两个预训练目标来优化GLM，两个目标交替进行：  

1. 文档级别：从文档中随机采样一个文本片段进行掩码，片段的长度为文档长度的50%-100%。
2. 句子级别：从文档中随机掩码若干文本片段，每个文本片段必须为完整的句子，被掩码的词数量为整个文档长度的15%。

# 四. Others
1. encoder和decoder共享参数
2. 随机打散被mask的片段，为了：完整的捕捉到不同片段之间的依赖关系
# 五. Code
## Generation
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
model = model.half().cuda()
model.eval()

# Inference
inputs = tokenizer("Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.", return_tensors="pt")
inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
inputs = inputs.to('cuda')
outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
print(tokenizer.decode(outputs[0].tolist()))

# Training
inputs = tokenizer(
    ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"],
    return_tensors="pt", padding=True)
inputs = tokenizer.build_inputs_for_generation(inputs, targets=["Beijing", "No"], max_gen_length=8, padding=False)
inputs = inputs.to('cuda')
outputs = model(**inputs)
loss = outputs.loss
logits = outputs.logits
```
## Classification
```
from transformers import AutoTokenizer, AutoModelForMultipleChoice
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
model = AutoModelForMultipleChoice.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
model = model.half().cuda()
model.eval()

inputs = tokenizer(["Tsinghua University is located in [MASK].",
                    "One minus one equals zero, is it correct? Answer: [MASK]"], return_tensors="pt", padding=True)
choices = [["Beijing", "Shanghai"], ["Yes", "No"]]
inputs = tokenizer.build_inputs_for_multiple_choice(inputs, choices)
inputs = inputs.to('cuda')
outputs = model(**inputs)
logits = outputs.logits
```
