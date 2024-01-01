1. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
   ```
   Using Uncertainty to Weigh Losses
   大loss的task给予小权重，小loss的task给予大权重。
   ```
   这篇论文指出了多任务学习模型的效果很大程度上由共享的权重决定，但训练这些权重是很困难。由此引出uncertainty的概念，来衡量不同的task的loss，使得可以同时学习不同类型的task。
2. GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks
   ```
   梯度正则化gradient normalization (GradNorm)，它能自动平衡多task不同的梯度量级，提升多任务学习的效果，减少过拟合。
   ```
   GradNorm设计了额外的loss来学习不同task loss的权重，但它不参与网络层的参数的反向梯度更新，目的在于不同task的梯度通过正则化能够变成同样的量级，使不同task可以以接近的速度进行训练    
3. End-to-End Multi-Task Learning with Attention  
<img width="516" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/0085a69f-00f6-4d35-ac07-beaeaf3f1e3d">  
4. A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation



--------------------------------------------------------------
https://zhuanlan.zhihu.com/p/456089764

   
