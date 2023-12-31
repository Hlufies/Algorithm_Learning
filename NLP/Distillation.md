# 模型蒸馏原理
```
Hinton在NIPS2014[1]提出了知识蒸馏（Knowledge Distillation）的概念，旨
在把一个大模型或者多个模型ensemble学到的知识迁移到另一个轻量级单模型上，方便部署。
```
**简单的说就是用小模型去学习大模型的预测结果，而不是直接学习训练集中的label** 

```
在蒸馏的过程中，
我们将原始大模型称为教师模型（teacher），
新的小模型称为学生模型（student），
训练集中的标签称为hard label，
教师模型预测的概率输出为soft label，
temperature(T)是用来调整soft label的超参数。
```

**蒸馏这个概念之所以work，核心思想是因为好模型的目标不是拟合训练数据，而是学习如何泛化到新的数据。所以蒸馏的目标是让学生模型学习到教师模型的泛化能力，理论上得到的结果会比单纯拟合训练数据的学生模型要好。**
