![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/57cc8be3-a640-42de-a076-1f9c47a380dc)# 扩散过程  
扩散模型包含两个过程：前向扩散过程和反向生成过程，前向扩散过程是对一张图像逐渐添加高斯噪音直至变成随机噪音，而反向生成过程是去噪音过程  
# Diffusion 和 VAE的区别  
从我的理解，Diffusion和VAE是从两种不同的角度进行建模。VAE是压缩信息的角度，建立隐变量模型，其编码器和解码器是成对训练。而扩散模型则是从物理热力动力学的角度建模，利用信息论里面熵增熵减和马尔可夫链规则来建立同纬度的隐变量模型。  
# noise schedule  
预先设置好的variance schedule, 简单理解就是构建一组方差系数，从小变大，原始数据而变成了一个随机噪音。  
# 重参数技巧
在扩散过程中，我们是对于根据马尔可夫链进行建模，即当前步依赖于上一步，而在每一步过程中随机采样相当于有不可避免的信息损失。  
例如：在传统的VAE中，通过对潜在变量进行随机采样来生成新的样本。然而，这种随机采样操作无法直接参与反向传播的梯度计算，因为采样操作是不可导的。为了解决这个问题，重参数化技巧被引入。  
重参数化技巧通过将采样操作与可导操作相结合，将随机性从采样过程中移除，使得采样操作的梯度计算成为可能。具体而言，在重参数化技巧中，潜在变量被表示为通过一个确定性的可导函数对一个固定的噪声源进行变换得到的结果。这样，潜在变量的采样过程可以分解为两个步骤：首先，从固定的噪声源中采样；然后，通过确定性的可导函数对这个采样结果进行变换。这种变换过程可以被包含在计算图中，并且可以通过反向传播来计算梯度。重参数化技巧使得在训练VAE时可以对潜在变量进行采样，并通过反向传播来更新模型参数，从而实现对潜在表示的学习。它提供了一种有效的训练方法，使得VAE能够生成具有多样性的样本，并且可以通过调整潜在变量来控制生成样本的特征。除了在VAE中的应用，重参数化技巧还可以在其他涉及采样操作的模型中发挥作用，例如生成对抗网络（GANs）中。通过将随机性的采样操作与可导的操作相结合，重参数化技巧为这些模型的训练提供了一种有效且可优化的方式。  

在扩散模型中，从xt - xt-1的这一步过程中，为了使得可以反向传播可导，因此引入符合高斯分布的e， 使得x = u + be, 将x的不确定性转移到e里面了。于是，从原来对x的不可导变为可导

# 反向过程  
反向过程实际是一个去噪过程。也就是要求真实分布q(xt-1|xt).而在这里，我们利用神经网络来学习每一步去噪的过程。我们将反向过程也定义为一个马尔卡夫链，只不过它是由一系列用神经网络参数化的高斯分布来组成。公式上优化，对于真实分布加入x0条件，使其可以拆解  
# 变分推断  
variational lower bound（VLB，又称ELBO）作为最大化优化目标，这里有
![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/f0ac6f74-ee5e-409b-94a9-49b0ee16c52f)  
![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/f529e91e-a2a5-4cdd-be2a-487e04b0383a)  

![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/c1c174a0-5c34-494e-8c23-96882eeebc89)
![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/0afb34e0-7bee-4bf2-a19f-398525156a2b)


![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/9e0a3043-fa9b-4516-829a-9ab0b56d7be0)  
![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/bc20c24b-0216-4e3c-b703-e03bb1b0abbf)



# 模型  
前面我们介绍了扩散模型的原理以及优化目标，那么扩散模型的核心就在于训练噪音预测模型，由于噪音和原始数据是同维度的，所以我们可以选择采用AutoEncoder架构来作为噪音预测模型。DDPM所采用的模型是一个基于residual block和attention block的U-Net模型。如下所示：  
U-Net属于encoder-decoder架构，其中encoder分成不同的stages，每个stage都包含下采样模块来降低特征的空间大小（H和W），然后decoder和encoder相反，是将encoder压缩的特征逐渐恢复。U-Net在decoder模块中还引入了skip connection，即concat了encoder中间得到的同维度特征，这有利于网络优化。DDPM所采用的U-Net每个stage包含2个residual block，而且部分stage还加入了self-attention模块增加网络的全局建模能力。 另外，扩散模型其实需要的是TTT个噪音预测模型，实际处理时，我们可以增加一个time embedding（类似transformer中的position embedding）来将timestep编码到网络中，从而只需要训练一个共享的U-Net模型。具体地，DDPM在各个residual block都引入了time embedding，如上图所示。  

# 为什么DDPM一定要这么多次采样  
# 为什么非要一步一步降噪，跳步行不行？  
答案是不行。注意，我们的优化目标虽然是噪声 ϵ\epsilon\epsilon 的MSE，但实际这只是重参数化（reparameterization）的结果。我们的优化终极目标实际上是去拟合概率分布 P(xt−1|x0,xt)P(x_{t-1}|x_0,x_t)P(x_{t-1}|x_0,x_t) 。而我们求得这个分布均值的方法是用的贝叶斯公式。 也就是在前面写到的，

![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/5f307ff2-d979-41c3-b70e-f30a68a1ab79)
![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/500c342c-00f0-4733-8caf-07a3b9920357)

# 如果我们预测出了 e， 直接得出x可以吗？
不行，答案同上面那一个问题。

ok，既然不能跳步的原因是由于马尔可夫性质的约束，那假设我不再让这个过程是马尔可夫的（Non-Markovian）了，现在我可不可以跳步了？答案是可以，并且这就是DDIM所做的事。

# DDIM : DDIM是如何去马尔可夫化的    
![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/315f1600-92c1-4d98-a3de-65fc14cfc1fc)

![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/e8b4a496-8b9a-495b-be2d-e6151f9b24ab)

![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/fe6e937b-6f3c-47de-b583-31e6966e7aa2)







