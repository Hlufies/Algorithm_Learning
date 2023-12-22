<p align="center">
<img width="554" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/f1fb9bc7-fa27-49c7-ad9a-0fbf737e512e"> 
</p>

## Github链接：https://github.com/TianxingWu/FreeInit  
<img width="702" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/e53a030c-03b1-4c73-a073-d6df08a4191c">  

图 1. 用于视频生成的 FreeInit。我们提出了 FreeInit，这是一种简洁而有效的方法，可以显着提高扩散模型生成的视频的时间一致性。 FreeInit 不需要额外的训练，也不需要引入可学习的参数，并且可以在推理时轻松地合并到任意视频扩散模型中。  
## Abstract  
```
In this paper, we delve deep into the noise initialization of video diffusion models, and discover an implicit training-inference gap that attributes to the unsatisfactory inference quality.
在本文中，我们深入研究了视频扩散模型的噪声初始化，并发现了隐含的训练推理差距，该差距归因于推理质量不佳。
我们的主要发现是：
1）推理时初始潜在的时空频率分布与训练时的时空频率分布本质上不同，
2）去噪过程受到初始噪声的低频分量的显着影响。
```
```
受这些观察的启发，我们提出了一种简洁而有效的推理采样策略 FreeInit，它显着提高了扩散模型生成的视频的时间一致性。
通过迭代细化推理过程中初始潜在变量的时空低频分量，FreeInit 能够补偿训练和推理之间的初始化差距，从而有效提高生成结果的主体外观和时间一致性。
大量实验表明，FreeInit 能够持续增强各种文本到视频生成模型的生成结果，而无需额外训练。
```



