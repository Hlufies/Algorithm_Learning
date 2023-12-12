# 彭森1·陈宇飞1·徐杰1·陈子卓1·王聪1·贾晓华1
# Department of Computer Science, City University of Hong Kong, Tat Chee Avenue, Hong Kong, Hong Kong SAR, China

# Abstract
```
深度学习已广泛应用于解决许多任务，例如图像识别、语音识别和自然语言处理。
训练大规模深度神经网络（DNN）模型需要高质量的数据集、先进的专家知识和巨大的计算能力，这使其具有足够的价值，可以作为知识产权（IP）受到保护。
保护DNN模型免受非法使用、复制和复制等知识产权侵犯对于深度学习技术的健康发展尤为重要。人
们已经开发了许多方法来保护 DNN 模型 IP，
```
***例如 DNN 水印、DNN 指纹、DNN 身份验证和推理扰动。***
```
鉴于其重要性，DNN 知识产权保护仍处于起步阶段。
在本文中，我们对现有的 DNN IP 保护方法进行了全面的调查。
我们首先总结了DNN模型的部署模式并描述了DNN IP保护问题。
然后我们根据现有的保护策略对现有的保护方法进行分类并详细介绍。
最后，我们比较这些方法并讨论 DNN IP 保护的未来研究主题。
```
# Keywords
```
Machine learning · Deep neural network models · Artificial intelligence security · Intellectual property protection
```

# Introduction
Machine Learning as a Service (MLaaS)    
Black-Box   
White-Box  

1. Attack method: 模型微调，模型压缩（模型蒸馏）
2. Protect method
   ```
   a. 从带水印的模型生成的被盗模型继承了嵌入的水印，可以提取水印来证明 IP 违规。具体取决于水印是否嵌入到模型的静态内容（如其权重）或动态内容（如其功能）中
   静态：静态方法通常修改训练损失函数以改变模型权重的分布以嵌入水印。
   动态：动态方法倾向于基于[对抗性示例]或 [DNN 后门攻击]将水印构建为触发数据集。动态 DNN 水印利用了模型训练过程的过度参数化，影响了原始模型的功能。此外，最近的研究[30-34]指出，大多数现有的水印方法仍然容易受到水印去除攻击。
   ```
