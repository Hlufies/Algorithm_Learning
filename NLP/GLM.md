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
<img width="574" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/fb8a8cdd-c712-4aa5-8bbc-40ecbba2a62a">

### 3.1.2 Multi-Task Pretraining

