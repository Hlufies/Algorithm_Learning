# Question   
1. 图文多模态预训练的时候，我们往往侧重于图文理解，而没有一个统一的框架来完成理解任务和生成任务，即understanding task and generation task；  
2. 图文多模态预训练往往需要大量的图文对样本，而高质量的图文对数据十分稀少，网上爬取的图文对数据包含较多噪声，这些尚未对齐的图文对样本对预训练模型是不利的；

# Solution
1. 提出一个多模态混合模型multimodal mixture，即包含encoder，又包含decoder，能够同时完成图文理解任务和文本生成任务；  
2. 提出Captioner-Filter的训练方式，从大量的带有噪声的图文对数据中过滤出对齐的图文对样本，从而进一步增强预训练模型的效果；

# Framework
1. BLIP框架是一种encoder-decoder混合多模态网络，主要包含三个部分，分别对应了三种任务。  
2. 图文对比任务ITC，图像encoder分支采用ViT结构，文本encoder部分采用BERT结构，[CLS] token用来代表整个句子的特征。图像特征和文本特征进行对比损失，从而进行对齐操作。  
3. 图文匹配任务ITM，采用image-grounded text encoder，具体结构也是BERT，采用cross-attention来融合图像特征和文本特征，相当于一个多模态的encoder，[Encode] token用来代表多模态融合后的特征，最后在该特征后面增加一个mlp分类head，区分图文是否匹配。  
4. 语言模型任务LM，采用image-grounded text decoder，采用causal self-attention来编码图像特征和文本特征。与BERT的mask机制不同，这里causal self-attention是一种序列注意力机制，相当于预测接下来的语句。[Decode] token被用来表示句子的开头。因此该语言模型任务能够为图像生成一段话caption，为后续的captioner、filter机制奠定基础。

# Train
1. 先利用网络上获取的图文对和人工标注好的图文对，对BLIP网络进行预训练，获得一个基础版本的多模态混合encoder-decoder模型。
2. 然后利用ITC和ITM任务，在人工标注的高质量图文对上，对BLIP的image-grounded text encoder进行finetune，获得一个高置信度的Filter。
3. 然后利用LM任务，在人工标注的高质量图文对上，对BLIP的image-grounded text decoder进行finetune，获得一个高质量的Captioner。
4. 然后针对网络获取的图文对中的图片进行captioning操作，获取描述文本，与图片组成一个图文对。将该图文对和网络获取的图文对一起送进filter中，获取过滤成功的图文对。
5. 最后将过滤成功的图文对，和人工标注的高质量图文对组成一个全新的数据集，利用该数据集对BLIP进行预训练，获得更加高质量的图文预训练模型。  

https://openatomworkshop.csdn.net/6645ac13b12a9d168eb6c2d6.html?dp_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6MTU0NDM5NywiZXhwIjoxNzE3OTgzOTA5LCJpYXQiOjE3MTczNzkxMDksInVzZXJuYW1lIjoicXFfNDU3NjEzOTIifQ.gy3pFEH0lRfLHrIF7OTDJduCTb3ZxvhSJGIkmtPwgAg
