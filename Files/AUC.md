1. 混淆矩阵的定义
```
(1) 四个区域TP、FP、TN、FN，其前面的T或者F代表的是True或者False，表示预测对了还是预测错了。
(2) P或者N表示正样本或者负样本，也就是预测的结果。那么TP就表示预测对了的正样本，也就是真实情况是正样本，预测也是正样本的这一部分。
(3) FN表示预测错误的负样本，也就是真实情况是正样本，但是预测成了负样本的这一部分数据。
```
![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/ca2cef14-e5b9-4a64-964d-c848d275ad66)

3. 基于混淆矩阵的相关指标定义
```
P   精确率: TP/(TP+FP)    意义为：在所有预测为1的样本中，label确实为1的占比
R   召回率: TP/(TP+FN)    意义为：对于所有label为1的样本中，实际预测也为1的数量比，或者说我们的模型或策略可以召回多少正样本
Acc 准确率：(TP+TN)/(P+R) 意义为：模型或策略预测正负样本都正确的数量占总数的比例
Err 错误率：1 - Acc       意义为：模型或策略预测正负样本都正确的数量占总数的比例
TRP 真阳率：TP/(TP+FN)    意义为：等同recall
FRP 假阳率：FP/(FP+TN)    意义为：真实标签是-1的数据，也就是所有负样本中，被预测成正样本的数量占比
```
4. PR曲线
```
我们看分子，精度和召回率的计算分子都是TP，也就是预测正确的正样本的数量。
同时，从宏观上看，精度和召回率也都是描述正样本预测好坏的指标。所以我们说，PR曲线更加关心和能反映出模型对正样本的预测能力。
PR曲线的两个坐标轴即Recall、Precision分别表示模型对于正样本的查全率和查准率。也就是说PR曲线能够反应正样本的预测状况
```
```
?????
#y_true为样本实际的类别，y_scores为样本为正例的概率
y_true = np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0])
y_scores = np.array([0.9, 0.75, 0.86, 0.47, 0.55, 0.56, 0.74, 0.62, 0.5, 0.86, 0.8, 0.47, 0.44, 0.67, 0.43, 0.4, 0.52, 0.4, 0.35, 0.1])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

precision [0.78571429 0.75081818182 0.8 0.77777778 0.875 0.85714286 1. 1. 1. 1. 1.1]
recall    [0.81818182 0.81818182 0.72727273 0.63636364 0.63636364 0.54545455 0.54545455 0.45454545 0.36363636 0.27272727 0.090909090.]
th        [0.47 0.5 0.52 0.55 0.56 0.62 0.67 0.74 0.75 0.8 0.86 0.9 ]
```
6. ROC曲线定义及其绘制
```
ROC曲线和PR曲线类似，都可以用来描述分类效果的好坏。但ROC和PR不同的是，ROC的横坐标假阳率FPR，纵坐标是真阳率TPR也就是Recall。
ROC的绘制并不需要设置阈值，将输出的概率转为1或者0。因为引入这样的阈值其实本质上是引入了一个超参，此参数的调节也会影响最终的结果。所以ROC的绘制步骤如下
(1)首先将所有样本按照预测概率的大小进行排序
(2)以每条样本的预测概率为阈值（而不是引入阈值），例如当前样本的预测概率为0.2，那么小于0.2都为负样本，大于等于0.2为正样本。
(3)在当前阈值的情况下，计算TPR、FPR，标记该坐标点。
(4)选取顺序中的下一个样本，执行2操作，直到所有样本全部执行完毕
```
```
y_true = [0,0,1,0,1,0,1,1,1,0,1,0]
y_pred = [0.1,0.3,0.34,0.38,0.45,0.5,0.55,0.6,0.7,0.75,0.8,0.9]

fpr,tpr,threshold = roc_curve(y_true,y_pred)
```
8. ROC曲线和PR曲线比较
9. AUC及其意义
10. 如何计算AUC
11. python实现AUC的计算
12. AUC面试题
```
优点：

AUC衡量的是一种排序能力，因此特别适合排序类业务；
AUC对正负样本均衡并不敏感，在样本不均衡的情况下，也可以做出合理的评估。
其他指标比如precision，recall，F1，根据区分正负样本阈值的变化会有不同的结果，而AUC不需要手动设定阈值，是一种整体上的衡量方法。

缺点（字节面试问到了AUC的缺点，一时竟然答不上来...）：

忽略了预测的概率值和模型的拟合程度；
AUC反映了太过笼统的信息。无法反映召回率、精确率等在实际业务中经常关心的指标；
它没有给出模型误差的空间分布信息，AUC只关注正负样本之间的排序，并不关心正样本内部，或者负样本内部的排序，这样我们也无法衡量样本对于好坏客户的好坏程度的刻画能力；
```
14. MapReduce实现AUC的计算