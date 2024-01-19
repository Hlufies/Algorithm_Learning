# 1. BPE:即字节对编码。
其核心思想是从字母开始，不断找词频最高、且连续的两个token合并，直到达到目标词数。
# 2. BBPE
BBPE核心思想将BPE的从字符级别扩展到子节(Bvte)级别BPE的一个问题是如果遇到了unicode编码，基本字符集可能会很大。BBPE就是以一个字节为一种“字符”，不管实际字符集用了几个字节来表示一个字符。这样的话，基础字符集的大小就锁定在了256(2^8)采用BBPE的好处是可以跨语言共用词表，显著压缩词表的大小。而坏处就是，对于类似中文这样的语言，一段文字的序列长度会显著增长因此，BBPE based模型可能比BPE based模型表现的更好。然而BBPE sequence比起BPE来说略长，这也导致了更长的训练/推理时间。BBPE其实与BPE在实现上并无大的不同，只不过基础词“-56的字节集。
# 3. WordPiece
WordPiece算法可以看作是BPE的变种。不同的是WordPiece基于概率生成新的subword而不是下一最高频字节对。WordPiece算法也是每次从词表中选出两个子词合并成新的子词。BPE选择频数最高的相邻子词合并，而WordPiece选择使得语言模型概率最大的相邻子词加入词表。
# 4. Unigram
它和 BPE 以及 WordPiece 从表面上看一个大的不同是，前两者都是初始化一个小词表，然后一个个增加到限定的词汇量，而Unigram Language Model 却是先初始一个大词表，接着通过语言模型评估不断减少词表，直到限定词汇量。
# SentencePiece
SentencePiece是把一个句子看作一个整体，再拆成片段，而没有保留天然的词语的概念。一般地，它把空格也当作一种特殊字符来处理，再用BPE或者Unigram算法来构造词汇表。SentencePiece除了集成了BPE、ULM子词算法之外，SentencePiece还能支持字符和词级别的分词。
