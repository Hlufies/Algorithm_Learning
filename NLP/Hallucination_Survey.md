<img width="667" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/31cf0d52-cb93-44bd-ba4b-2fe1a9234b04">  

# Abstract
随着大型语言模型（LLM）在编写类似人类文本的能力方面不断进步，一个关键的挑战仍然围绕着它们的“幻觉”倾向——生成看似事实但毫无根据的内容。  
# Introduction
大型语言模型（LLM）中的幻觉需要**创建跨越多个主题的事实上错误的信息**。例如，法学硕士的一个基本问题是他们容易产生有关现实世界主题的错误或捏造的细节。这种提供不正确数据的倾向（通常称为幻觉）对该领域的研究人员提出了重大挑战。它会导致 GPT-4 等高级模型和其他同类模型可能生成不准确或完全没有根据的参考。   
出现此问题的原因是训练阶段的**模式生成技术和缺乏实时互联网更新，导致信息输出存在差异**。
## 缓解技术
在当代计算语言学中，减轻幻觉是一个关键焦点。研究人员提出了各种策略来应对这一挑战，包括：  
1. 反馈机制、
2. 外部信息检索
3. 语言模型生成的早期改进
本文通过将这些不同的技术整合和组织成一个综合的分类法而具有重要意义。从本质上讲，本文对法学硕士幻觉领域的贡献有三个方面：   
3.1 Introduction of a systematic taxonomy designed to categorize hallucination mitigation techniques for LLMs, encompassing Vision Language Models (VLMs).  
3.2 Synthesis of the essential features characterizing these mitigation techniques, thereby guiding more structured future research endeavors within this domain.  
3.3 Deliberation on the limitations and challenges inherent in these techniques, accompanied by potential solutions and proposed directions for future research.  
<img width="620" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/f0ffc51e-905a-4096-919c-70efdea26b3b">

 法学硕士中幻觉缓解技术的分类，重点关注涉及**模型开发和提示技术**的流行方法。模型开发分为多种方法，包括新的解码策略、基于知识图的优化、添加新颖的损失函数组件和监督微调。同时，提示工程可以涉及基于检索增强的方法、基于反馈的策略或提示调整。

 
