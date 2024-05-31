TN TP FN FP
accuracy = TN/(TN+TP+FN+FP)

precision = TP/(TP+FP)

recall = TP/(TN+TP)

F1 = P*R*2/(P+2)

Precision和Recall是评估分类模型性能的重要指标，但它们各自存在一定的问题和局限性。

### Precision（精确率）的问题

1. **忽略了假阴性**：
   - Precision只考虑了预测为正类的样本中有多少是真正的正类样本，而忽略了真正的正类样本中有多少被预测为负类（假阴性）。在某些应用场景下，这可能会导致对假阴性样本的忽视。
   - 例如，在医疗诊断中，假阴性意味着将患病的病人误诊为健康，这可能带来严重后果。

2. **不适用于类不平衡数据**：
   - 当正类样本非常少而负类样本很多时，高Precision可能很容易达到，因为大部分样本都被预测为负类，从而减少了假阳性，但这并不代表模型整体性能好。

### Recall（召回率）的问题

1. **忽略了假阳性**：
   - Recall只考虑了实际正类样本中有多少被正确预测为正类，而忽略了被错误预测为正类的负类样本（假阳性）。这可能会导致对假阳性样本的忽视。
   - 例如，在垃圾邮件过滤中，假阳性意味着将正常邮件误判为垃圾邮件，用户可能因此错过重要信息。

2. **不适用于类不平衡数据**：
   - 在类不平衡的数据集中，高Recall可能意味着很多负类样本被误判为正类，增加了假阳性的数量，影响了实际使用效果。

### Precision和Recall共同的问题

1. **无法平衡两者**：
   - Precision和Recall往往是此消彼长的关系。提高Precision通常会降低Recall，反之亦然。单独依赖一个指标无法全面反映模型性能。
   - 例如，在某些应用中需要在Precision和Recall之间找到平衡点，但单独使用Precision或Recall无法达到这一目的。

2. **缺乏全局视角**：
   - Precision和Recall都只关注正类样本，忽略了负类样本的分类效果。它们不能全面反映模型在整个数据集上的表现。

### 综合解决方法

为了克服Precision和Recall的局限性，常用以下几种方法：

1. **F1-score**：
   - F1-score是Precision和Recall的调和平均数，综合考虑了两者，可以在不平衡数据集中提供更全面的评价。
   - \[
   F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
   \]

2. **ROC曲线和AUC**：
   - ROC曲线通过绘制真正率（True Positive Rate, TPR）与假正率（False Positive Rate, FPR）的关系来评估模型性能。AUC（Area Under Curve）是ROC曲线下的面积，反映了模型的整体性能。

3. **精度-召回曲线（Precision-Recall Curve）**：
   - 这种曲线可以更好地评估模型在不平衡数据集上的表现，通过观察不同阈值下的Precision和Recall的变化，找到最优平衡点。

### 总结

Precision和Recall各自有其优点和局限性，无法单独全面反映模型性能。为了更准确地评估和比较模型，常常需要结合使用多种指标，如F1-score、ROC曲线和AUC等。这些综合指标可以提供更全面的视角，帮助找到适合具体应用场景的最佳模型。  

当然，以下是Precision（精确率）、Recall（召回率）及其相关指标的公式：

### Precision（精确率）

精确率是指预测为正类的样本中实际为正类的比例。公式如下：

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

其中：
- \( TP \)（True Positive）：真正类，即被正确预测为正类的样本数。
- \( FP \)（False Positive）：假正类，即被错误预测为正类的负类样本数。

### Recall（召回率）

召回率是指实际为正类的样本中被正确预测为正类的比例。公式如下：

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

其中：
- \( TP \)（True Positive）：真正类，即被正确预测为正类的样本数。
- \( FN \)（False Negative）：假负类，即被错误预测为负类的正类样本数。

### F1-score

F1-score是Precision和Recall的调和平均数，综合考虑了两者。公式如下：

\[
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

### True Positive Rate（真正率，TPR）或Sensitivity（敏感度）

真正率（也称为敏感度或召回率）是指实际为正类的样本中被正确预测为正类的比例。公式如下：

\[
\text{TPR} = \text{Recall} = \frac{TP}{TP + FN}
\]

### False Positive Rate（假正率，FPR）

假正率是指实际为负类的样本中被错误预测为正类的比例。公式如下：

\[
\text{FPR} = \frac{FP}{FP + TN}
\]

其中：
- \( TN \)（True Negative）：真负类，即被正确预测为负类的样本数。

### ROC曲线（Receiver Operating Characteristic Curve）

ROC曲线是通过绘制TPR（真正率）与FPR（假正率）的关系来评估模型性能。曲线下的面积称为AUC（Area Under Curve），表示模型的整体性能。

### Precision-Recall曲线

Precision-Recall曲线通过绘制不同阈值下的Precision和Recall的关系来评估模型性能，尤其适用于不平衡数据集。

这些公式和曲线结合使用，可以更全面地评估分类模型的性能，帮助找到适合具体应用场景的最佳模型。
