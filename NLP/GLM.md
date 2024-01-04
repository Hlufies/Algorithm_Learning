# 一. Conlusion 
1. GLM -> **autoregressive blank infilling** -> **advantage**: TO solve the problem of NLU AND NLG 
2. autoregressive: GPT: left->right language model -> **advantage**: long-text generation and show fewshot learning ability -> **disadvantage** is the unidirectional attention mechanism, which **cannot fully capture the dependencies between the context words in NLU tasks**.
3. autoencoding: BERT -> learn bidirectional context encoders via denoising objectives, e.g.**Masked Language Model (MLM)** -> **advantage**: suit natural language understanding tasks -> **disadvantage**: could not be directly applied for text generation(NLG).
4. encoder-decoder: T5 -> adopt bidirectional attention for the encoder, unidirectional attention for the decoder, and cross attention between them -> conditional generation tasks -> text summarization and response generation
# 二. Improvement 
1. span shuffling
2. 2D positional encoding.
# 三. Pretraining Framework
我们提出了一种基于新颖的自回归空白填充目标的通用预训练框架 GLM。 GLM 将 NLU 任务制定为包含任务描述的完形填空问题，可以通过自回归生成来回答。
## 3.1 Pretraining Objective
### 3.1.1 
