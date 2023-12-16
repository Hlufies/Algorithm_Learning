# 思维链CoT(Chain Of Thought Prompting)
```
思维链(CoT)提示过程1是一种最近开发的提示方法，它鼓励大语言模型解释其推理过程。
下图显示了 few shot standard prompt（左)与链式思维提示过程（右）的比较。
```
<img width="580" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/fa85d652-9764-4934-a66b-2e55e6cd1a82">

# 零样本思维链(Zero Shot Chain of Thought，Zero-shot-CoT)
完整的零样本思维链过程涉及两个单独的提示/补全结果  
<img width="399" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/a1a07abf-16cc-4ef5-bfbc-7326eabbccf3">
```
Kojima等人尝试了许多不同的零样本思维链提示（例如“让我们按步骤解决这个问题。”或“让我们逻辑思考一下。”），但他们发现“让我们一步一步地思考”对于他们选择的任务最有效。
提取步骤通常必须针对特定任务，使得零样本思维链的泛化能力不如它一开始看起来的那样强。
从个人经验来看，零样本思维链类型的提示有时可以有效地提高生成任务完成的长度。
例如，请考虑标准提示写一个关于青蛙和蘑菇成为朋友的故事。在此提示的末尾附加让我们一步一步地思考会导致更长的补全结果。
```
# Self-Consistency
<img width="573" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/31db92f3-d5e4-4489-831c-f30b7d49c868">
```
研究表明，自洽性可以提高算术、常识和符号推理任务的结果。
即使普通的思路链提示被发现无效2，自洽性仍然能够改善结果。
Wang 等人讨论了一种更复杂的边缘化推理路径方法，该方法涉及每个思路链生成的大语言模型概率。
然而，在他们的实验中，他们没有使用这种方法，多数投票似乎通常具有相同或更好的性能。
```

# Generated Knowledge Approach
生成的知识方法（Generated Knowledge Approach）1要求LLM 在生成响应之前生成与问题相关的可能有用的信息。
该方法由两个中间步骤组成，即**知识生成**和**知识集成**。
<img width="561" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/dad8bf98-c292-4d7f-870f-219756455073">
```
结论
这种方法显示了对各种常识数据集的改进。
备注
The knowledge corresponding to the selected answer is called the selected knowledge. 与所选答案对应的知识称为“精选知识”。
```

# 最少到最多提示过程
```
最少到最多提示过程 (Least to Most prompting, LtM)1 将思维链提示过程 (CoT prompting) 进一步发展，首先将问题分解为子问题，然后逐个解决。
它是受到针对儿童的现实教育策略的启发而发展出的一种技术。
与思维链提示过程类似，需要解决的问题被分解成一组建立在彼此之上的子问题。在第二步中，这些子问题被逐个解决。
与思维链不同的是，先前子问题的解决方案被输入到提示中，以尝试解决下一个问题。
```
<img width="512" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/f55e9148-82fd-4ba1-b9bb-1252000ea440">

```
结论
LtM 带来了多项提升：

相对于思维链提高了准确性
在难度高于提示的问题上提升了泛化能力
在组合泛化方面的性能得到了显著提高，特别是在SCAN基准测试3中
使用 text-davinci-002（论文中使用的模型）的标准提示解决了 6% 的 SCAN 问题，而 LtM 提示则取得了惊人的 76% 的成功率。在 code-davinci-002 中，结果更为显著，LtM 达到了 99.7% 的成功率。
```

# Dealing With Long Form Content  
  1. Processing the Text
    ```
    Removing unnecessary sections or paragraphs that are not relevant or contribute to the main message. This can help to prioritize the most important content.
    Summarizing the text by extracting key points or using automatic summarization techniques. This can provide a concise overview of the main ideas.
    ```
  2. Chunking and Iterative Approach
    ```
    可以将其划分为更小的块或部分，而不是一次将整个长格式内容提供给模型。
    这些块可以单独处理，允许模型一次专注于特定的部分。
    可以采用迭代方法来处理长格式内容。该模型可以为每个文本块生成响应，生成的输出可以作为下一个块的输入的一部分。
    这样，与语言模型的对话可以以循序渐进的方式进行，有效地管理对话的长度。
    ```
  3. Post-processing and Refining Responses
    ```
    模型生成的初始响应可能很长或包含不必要的信息。对这些响应执行后处理以细化和浓缩它们是很重要的。
    一些后处理技术包括：
    a. 删除多余或重复的信息。
    b. 提取响应中最相关的部分。
    c. 重新组织应对措施，以提高清晰度和连贯性。
    通过细化响应，可以使生成的内容更加简洁易懂。
    ```
  4. Utilizing AI assistants with longer context support
  5. Code libraries
  6. Conclusion
  [LangChain](https://github.com/langchain-ai/langchain)
  ```
  处理长格式内容可能很有挑战性，但通过采用这些策略，您可以在语言模型的帮助下有效地管理和浏览内容。记住要试验、迭代和完善您的方法，以确定最有效的策略来满足您的特定需求。
  ```


