通过对一些训练好的Transformer 模型中的注意力矩阵进行分析发现，其中很多通常是稀疏的，因此可以通过限制Query-Key对的数量来减少计算复杂度。这类方法就称为稀疏注意力（Sparse Attention）机制。可以将稀疏化方法进一步分成两类：基于位置信息和基于内容。  
![image](https://github.com/Hlufies/Algorithm_Learning/assets/130231524/181e2946-39dd-40fb-ac2a-4c2c43c75f99)

