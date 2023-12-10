# Multi-Class is not Multi-Label

```
文本分类是NLP的一项基础任务，属于自然语言理解，旨在对于给定文本文档，自动地分配正确的标签。
文本分类在许多方面的应用很多，例如：信息检索、自然语言推理、情感分析、问答等。
文本分类任务从分类目的上可以划分为三类：二元分类、多类别分类以及多标签分类。
```

# Multi-Class
```
在分类问题中，我们已经接触过二分类和多分类问题了。所谓二（多）分类问题，指的是y值一共有两（多）个类别，每个样本的y值只能属于其中的一个类别。
```
## 二分类
```
介绍：主要应用于情感分析，即对于特定文本，分析出文本的情感导向（喜欢/厌恶）。对于该情况，通常来说，分类器只需输出正类/负类的标签即可。
应用：情感分析、垃圾邮件分类。
```
## 多分类
```
介绍：相比较于二元分类，多类别分类 通常需要在多种类别（>2）的标签集中选出正确的标签分配给特定文本。同时，多类别分类要求每个文本也是只有一个标签。
```
# Multi-Label
```
Multi-Label，不同于多类别分类，多标签分类的总标签集合大，而且每个文本都包含多种标签，即将多个标签分配给特定文本。由于不同文本分配的标签集不同，给分类任务带来一定程度的困难。
另外，当总的标签集合数目特别大的时候，这种情况可以算作为一种新的多标签分类任务，即极端多标签文本分类（Extreme multi-label text classification (XMTC)）。
```
## Multi-Class vs Multi-Label
```
多分类任务中一条数据只有一个标签，但这个标签可能有多种类别。比如判定某个人的性别，只能归类为"男性"、"女性"其中一个。再比如判断一个文本的情感只能归类为"正面"、"中面"或者"负面"其中一个。
多标签分类任务中一条数据可能有多个标签，每个标签可能有两个或者多个类别（一般两个）。例如，一篇新闻可能同时归类为"娱乐"和"运动"，也可能只属于"娱乐"或者其它类别。
```
## Multi-Label 两大难点
```
问题1：类别的长尾分布（类别不平衡）：小规模的标签子集（head labels）拥有大量的样本，而绝大部分标签（tail labels）只拥有较少的样本；
方法1：重采样和重新加权
问题2：类别标签的联动（类别共现）：一些head labels会与tail labels同时出现；
方法1会导致 公共标签的过采样
```

## Multi-Label 损失函数 构建与选择
### BCE
```
存在问题：由于 head classes的主导以及negative instances的影响，导致 BCE Loss 函数 容易受到 类别不均衡问题 影响；
优化方向：绝大部分balancing方法都是reweight BCE从而使得稀有的instance-label对能够得到得到合理的“关注”
```
```
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELosswithLogits(nn.Module):
    def __init__(self, pos_weight=1, reduction='mean'):
        """
        初始化二元交叉熵损失函数

        参数:
        - pos_weight (float, optional): 正样本的权重，用于调整正样本的重要性，默认为1。
        - reduction (str, optional): 损失的降维方式，可选值为 'mean' 或 'sum'，默认为 'mean'。

        注意:
        - 当 reduction 为 'mean' 时，返回损失的平均值。
        - 当 reduction 为 'sum' 时，返回损失的总和。
        """
        super(BCELosswithLogits, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, target):
        """
        计算二元交叉熵损失

        参数:
        - logits (torch.Tensor): 模型输出的logits，形状为 [N, *]，N是样本数。
        - target (torch.Tensor): 真实标签，形状与logits相同，元素取值为0或1。

        返回:
        - loss (torch.Tensor): 二元交叉熵损失。
        """
        # 使用 sigmoid 函数将logits转换为概率
        logits = F.sigmoid(logits)

        # 计算二元交叉熵损失
        loss = - self.pos_weight * target * torch.log(logits) - (1 - target) * torch.log(1 - logits)

        # 根据reduction参数选择降维方式
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

```
