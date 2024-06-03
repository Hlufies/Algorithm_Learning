1. 介绍一下Diffusion 网络架构
2. 从ODE角度理解Diffsuion （songyang paper）
3. 介绍一下LLAMA prefil+decode
   ```
      prefill 阶段：将用户输入的 prompts 生成 q、k、v，存入 KV Cache（为 decode 阶段缓存）。这一步计算并行好，是计算密集型 compute bound
      decode 阶段：由最新产生的 tokens 生成 q、k、v，计算它与之前所有 tokens 的 attention，这一步需要从 KV Cache 中读取前面所有 token 的 key、value，因此是内存密集型 memory bound。

      vLLM 提出了 Paged Attention 算法，将 attention 算法产生的连续的 key value 向量按照 block 进行组织和管理，以减少显存碎片。
      vLLM 还借鉴操作系统当中的虚拟内存和分页思想优化 Transformer 模型推理中产生的 KeyValue Cache，大大提高了显存当中 KV Cache 的利用效率。
      vLLM 是基于单机层面上设计，只能将 GPU 中的 block swap 到单机的内存当中。

   ```
   ```
     采样策略
     为什么要进行不同的采样策略
     进行不同的采样策略可以对生成文本的多样性和质量进行调控，以满足不同的需求和应用场景。通过选择不同的采样策略，可以平衡生成文本的多样性和质量。
     1. 贪婪采样适用于需要高准确性的任务，
     2. 温度采样
     3. Top-k采样
     4. Top-p 采样则可以在一定程度上增加生成文本的多样性，使得输出更加丰富和有趣。具体选择哪种采样策略取决于应用的需求和期望的输出效果。
   ```
      https://blog.csdn.net/qq_43243579/article/details/136331123
   
5. KV cache
6. 根号下dk
7. ROPE编码 Alibi编码
