# CRF模型和HMM和MEMM模区别？
```
相同点：MEMM、HMM、CRF 都常用于 序列标注任务；
不同点：
与 HMM 的区别：CRF 能够解决 HMM 因其输出独立性假设，导致其不能考虑上下文的特征，限制了特征的选择的问题；
与 MEMM 的区别：MEMM 虽然能够解决 HMM 的问题，但是 MEMM 由于在每一节点都要进行归一化，所以只能找到局部的最优值，同时也带来了标记偏见的问题，即凡是训练语料中未出现的情况全都忽略掉。
CRF ：很好的解决了这一问题，他并不在每一个节点进行归一化，而是所有特征进行全局归一化，因此可以求得全局的最优值。
```
# 为什么 CRF模型 会比 HMM 被普遍使用？

```
原因 1：CRF模型 属于 判别式模型，在 序列标注 任务上，效果优于 生成式模型；
原因 2：HMM 提出 齐次马尔可夫性假设 和 观测独立性假设，这两个假设过强，而 CRF 只需要满足 局部马尔可夫性就好，通过降低假设的方式，提升模型效果；
```

# Code
```
import numpy as np

class CRF(object):
    def __init__(self, V, VW, E, EW):
        '''
        :param V: 节点上的特征函数，称为状态特征
        :param VW: V对应的权值
        :param E: 边上的特征函数，称为转移特征
        :param EW: E对应的权值
        '''
        self.V = V  # 点分布表
        self.VW = VW  # 点权值表
        self.E = E  # 边分布表
        self.EW = EW  # 边权值表
        self.D = []  # Delta表，最大非规范化概率的局部状态路径概率
        self.P = []  # Psi表，当前状态和最优前导状态的索引表
        self.BP = []  # BestPath，最优路径
        return

    def Viterbi(self):
        '''
        条件随机场预测问题的维特比算法
        '''
        self.D = np.full(shape=(np.shape(self.V)), fill_value=.0)
        self.P = np.full(shape=(np.shape(self.V)), fill_value=.0)
        for i in range(np.shape(self.V)[0]):
            # 初始化
            if 0 == i:
                self.D[i] = np.multiply(self.V[i], self.VW[i])
                self.P[i] = np.array([0, 0])
                print('self.V[%d]=' % i, self.V[i], 'self.VW[%d]=' % i, self.VW[i], 'self.D[%d]=' % i, self.D[i])
                print('self.P:', self.P)
            # 递推求解布局最优状态路径
            else:
                for y in range(np.shape(self.V)[1]):  # delta[i][y=1,2...]
                    for l in range(np.shape(self.V)[1]):  # V[i-1][l=1,2...]
                        delta = 0.0
                        delta += self.D[i - 1, l]  # 前导状态的最优状态路径的概率
                        delta += self.E[i - 1][l, y] * self.EW[i - 1][l, y]  # 前导状态到当前状态的转移概率
                        delta += self.V[i, y] * self.VW[i, y]  # 当前状态的概率
                        print('(x%d,y=%d)-->(x%d,y=%d):%.2f + %.2f + %.2f=' % (i - 1, l, i, y, \
                                                                                 self.D[i - 1, l], \
                                                                                 self.E[i - 1][l, y] * self.EW[i - 1][
                                                                                     l, y], \
                                                                                 self.V[i, y] * self.VW[i, y]), delta)
                        if 0 == l or delta > self.D[i, y]:
                            self.D[i, y] = delta
                            self.P[i, y] = l
                    print('self.D[x%d,y=%d]=%.2f\n' % (i, y, self.D[i, y]))
        print('self.Delta:\n', self.D)
        print('self.Psi:\n', self.P)

        # 返回，得到所有的最优前导状态
        N = np.shape(self.V)[0]
        self.BP = np.full(shape=(N,), fill_value=0.0)
        t_range = -1 * np.array(sorted(-1 * np.arange(N)))
        for t in t_range:
            if N - 1 == t:  # 得到最优状态
                self.BP[t] = np.argmax(self.D[-1])
            else:  # 得到最优前导状态
                self.BP[t] = self.P[t + 1, int(self.BP[t + 1])]

        # 最优状态路径表现在存储的是状态的下标，我们执行存储值+1转换成示例中的状态值
        # 也可以不用转换，只要你能理解，self.BP中存储的0是状态1就可以~~~~
        self.BP += 1

        print('最优状态路径为：', self.BP)
        return self.BP


def CRF_manual():
    S = np.array([[1, 1],  # X1:S(Y1=1), S(Y1=2)
                  [1, 1],  # X2:S(Y2=1), S(Y2=2)
                  [1, 1]])  # X3:S(Y3=1), S(Y3=1)
    SW = np.array([[1.0, 0.5],  # X1:SW(Y1=1), SW(Y1=2)
                   [0.8, 0.5],  # X2:SW(Y2=1), SW(Y2=2)
                   [0.8, 0.5]])  # X3:SW(Y3=1), SW(Y3=1)
    E = np.array([[[1, 1],  # Edge:Y1=1--->(Y2=1, Y2=2)
                   [1, 0]],  # Edge:Y1=2--->(Y2=1, Y2=2)
                  [[0, 1],  # Edge:Y2=1--->(Y3=1, Y3=2)
                   [1, 1]]])  # Edge:Y2=2--->(Y3=1, Y3=2)
    EW = np.array([[[0.6, 1],  # EdgeW:Y1=1--->(Y2=1, Y2=2)
                    [1, 0.0]],  # EdgeW:Y1=2--->(Y2=1, Y2=2)
                   [[0.0, 1],  # EdgeW:Y2=1--->(Y3=1, Y3=2)
                    [1, 0.2]]])  # EdgeW:Y2=2--->(Y3=1, Y3=2)

    crf = CRF(S, SW, E, EW)
    ret = crf.Viterbi()
    print('最优状态路径为:', ret)
    return


if __name__ == '__main__':
    CRF_manual()

```
