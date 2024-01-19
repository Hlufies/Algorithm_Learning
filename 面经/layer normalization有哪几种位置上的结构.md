# pre-norm:
layer norm在残差连接之前。pre-norm的优势在于能够给靠前的层一个绿色通道，不需要对这部分参数正则，防止梯度消失/爆炸，使得训练更容易;
# post-norm:
layer norm在残差连接之后。post-norm在残差之后做归一化，对参数正则化的效果更强。
# sandwitch-LN:
layer norm在残差连接之前和之后都加。
