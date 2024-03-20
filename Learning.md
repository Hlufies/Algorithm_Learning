# AI基础
## 1. 机器学习/深度学习基础   
学习目标：主要了解机器学习/**深度学习**的底层原理，如损失函数，激活函数，网络结构等。  
学习任务：复现交叉熵损失函数计算，Relu函数计算，CNN卷积操作（主要就是学习一下常用的张量运算，可以用pytorch）  
[参考链接1: 李宏毅](https://www.bilibili.com/video/BV1ou411N7X3/?spm_id_from=333.337.search-card.all.click&vd_source=ee5d618c255e7677033d82f9c5a69af1)  
[参考链接2: 李沐](https://zh-v2.d2l.ai/)  
  
## 2. CV
学习目标：主要学习一些经典的卷积神经网络, 如ResNet， Unet， VGG等。   
学习任务：复现ResNet或则Unet  
论文链接：  
1. https://arxiv.org/abs/1512.03385
2. https://arxiv.org/abs/1505.04597
 
[参考链接1: ResNet](https://github.com/rishivar/Resnet-18)  
[参考链接2: Uent](https://github.com/milesial/Pytorch-UNet)

## 3. NLP
学习目标：主要学习一些经典的语言模型, 如RNN，LSTM等。   
学习任务：复现LSTM  
[参考链接](https://github.com/yangwohenmai/LSTM)

**[注]**: 1,2,3是一些关于深度学习基础，初学太深入得太细节，懂得大概的流程就行了

## 4. Transformer && Bert && GPT 
学习目标：Transformer、BERT和GPT都是自然语言处理（NLP）领域的重要模型，它们在文本处理任务中取得了巨大的成功。目前主流的大语言模型基本是基于Transformer架构，因此这个板块十分重要。
学习任务：复现Transformer，了解Transformer论文细节。
论文链接：
1. Transformer：["Attention is All You Need"](https://arxiv.org/abs/1706.03762)  
2. BERT：["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805)  
3. GPT：["Improving Language Understanding by Generative Pretraining"](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)  
[参考代码链接](https://github.com/Hlufies/Algorithm_Learning/tree/main/Code/Transformer)

## 5. Diffusion Model
学习目标：Diffusion是目前最前沿的生成视觉，是始于 2020 年所提出的 DDPM（Denoising Diffusion Probabilistic Model），仅在 2020 年发布的开创性论文 DDPM 就向世界展示了扩散模型的能力，在图像合成方面击败了 GAN，所以后续很多图像生成领域开始转向 DDPM 领域的研究。  
前置知识：
1. GAN：GAN是生成对抗网络的缩写。这是一种深度学习框架，由两个神经网络组成：生成器（Generator）和判别器（Discriminator）。生成器试图生成看起来像真实数据的样本，而判别器则尝试区分生成器生成的样本与真实数据之间的差异。这两个网络在训练过程中相互对抗，通过竞争提高彼此的性能。    
2. VAE：VAE是变分自编码器的缩写。它是一种生成模型，也是一种无监督学习方法。VAE通过将输入数据编码成潜在空间中的分布，并通过随机采样来生成新的数据样本。与普通的自编码器不同，VAE通过引入概率性潜在空间和损失函数来优化模型，使其能够更好地生成具有多样性的数据样本。  
3. Flow：Flow指的是一类模型，用于建模数据的概率分布。Flow模型通过一系列可逆变换将输入数据从简单的先验分布映射到复杂的后验分布。这些模型通常具有良好的生成性能，并且能够有效地进行密度估计和采样。
4. Diffusion：Diffusion是指一种用于生成模型和密度估计的方法。它通过一个动态的随机过程，逐步将噪声添加到初始样本，从而生成新的数据样本。
学习任务：了解学习Diffusion的数学原理和思想。  

论文链接：
以下是每个术语相关的代表性论文：

1. **GAN (Generative Adversarial Networks)**:
   - [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
2. **VAE (Variational Autoencoder)**:
   - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
3. **Flow**:
   - [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)

4. **Diffusion**:
   - [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
   - [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
 
**[注]**: 4，5是一些进阶的知识，属于AIGC领域前沿的研究热点。
# AI安全
