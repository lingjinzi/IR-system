### 运行

1. 首先看utils.Constants.py，选择你的配置

```python
DEBUG_MODE = False				# Debug模式开关

QUERY_MODEL_NAMES = ['bm25', 'word2vec']
# 2 表示两种方式的加权综合			
QUERY_MODEL_ID = 2								# 查询模型选择
```



2. test.py执行测试程序，读取query.txt中的查询，将结果输出到result.txt当中
3. man.py为主程序，启动检索系统，用户可输入查询，结果输出到标准输出



### 算法

BM25算法
$$
\text{BM25}(D, Q) = \sum_{i=1}^{n} \frac{{(k+1) \cdot f_i}}{{f_i + k \cdot (1 - b + b \cdot \frac{\lvert D \rvert}{{\text{avgDL}}})}} \cdot \log\left(\frac{{N - n_i + 0.5}}{{n_i + 0.5}}\right)
$$



- b 参数 ，b 默认 0.75（经验值），主要是对长文档做惩罚，如果不希望文档长度更大的相似度搜索更好，可以把 b 设置得更大，如果设置为 0，文档的长度将与分数无关。从下图可以看到，b=0时，L与分数无关，b=1时，L越大，分数打压越厉害。

- k 参数， 默认值 1.2，会影响词语在文档中出现的次数对于得分的重要性，如果希望词语出现次数越大，文档越相关，这个参数可以设置更大。


BM只考虑了term和doc的维度，对query 里term的频率没有考虑，BM25+（Best Matching 25 Plus）正是基于这一点，来改进BM25算法

$$
\text{BM25+}(D, Q) = \sum_{i=1}^{n} \frac{{(k_1+1) \cdot f_i \cdot (k_3+1) \cdot qf}}{{(f_i + k_1 \cdot (1 - b + b \cdot \frac{{\lvert D \rvert}}{{\text{avgDL}}})) \cdot (k_3 + qf)}} \cdot \log\left(\frac{{N - n_i + 0.5}}{{n_i + 0.5}}\right)
$$





结合词嵌入向量技术，使用Word2Vec模型

Word2Vec 是一种将词语映射到连续向量空间的词嵌入模型。Word2Vec 通过神经网络学习词语的向量表示，使得在向量空间中，语义相似的词语距离较近。

改进后的算法
$$
\text{BM25+}(D, Q) = \sum_{i=1}^{n} \frac{{(k_1+1) \cdot r_i \cdot (k_3+1) \cdot qr}}{{(r_i + k_1 \cdot (1 - b + b \cdot \frac{{\lvert D \rvert}}{{\text{avgDL}}})) \cdot (k_3 + qf)}}
$$



### Reference

BM25
https://www.cnblogs.com/xiaoqi/p/18003267/bm25

模型训练
https://blog.csdn.net/yangbindxj/article/details/123911869