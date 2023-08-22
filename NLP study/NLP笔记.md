[TOC]

# NLP

==本质是:概率的预测==

## word2vec(预测)

![image-20230809111028037](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230809111028037.png)

先介绍第一种skip gram模型,通过中心单词来预测上下文的单词

1. 我们会给定一个上下文的区间,比如下图,上下文的区间为2,即预测前后两个单词的概率
2. 对于每个单词的概率,我们采用上图的公式计算概率.   u代表上文文的单词,v代表中心单词,w代表上下文单词的集合.
3. 每一个单词概率为,该词的向量与中心词向量点乘,除以所有上下文词向量和中心词向量的点乘的和.    返回一个概率分布
4. 类似于softmax函数

![image-20230809111149295](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230809111149295.png)

**像这样,一个中心词和他的上下文组成的小整体,叫做window**

---



## word2vec(训练参数)

![image-20230809112654786](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230809112654786.png)

θ代表模型的所有参数,在一个很长很大的向量中,每个单词有d维度的词向量,有v个单词,而且,每个单词有两个词向量,其中一个代表中心词,另一个代表上下文.

整个θ有2\*v*d个参数,训练的目的在于找到这些最优参数

### 梯度下降法

![image-20230809120626176](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230809120626176.png)

取了一个负对数

p(ω`t+j`|ω`t`)代表概率,我们希望预测的概率较大,也就是预测结果和现实中的结果很接近,所以是越大越好

第一个Σ代表一共有T个单词的概率求和,第二个Σ代表每个单词的上下文区间m的概率求和

损失函数越小越好,所以前面加个负号

### 梯度下降法弊端

对于现实世界中庞大的语料库,对于每一个词向量都进行更新,是个非常巨大的运算量

SGD随机梯度下降可以解决此问题

![image-20230809155431323](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230809155431323.png)

大多数情况下,一个词向量是一个行向量

在更新的过程中,我们只是更新特定行的词向量

### softmax函数弊端

对于一个词向量而言,计算概率也是一个挺大计算开销,因为其中涉及到累加操作.![image-20230809170858811](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230809170858811.png)

而且只是考虑了正类样本(实际中正确的,应该出现的单词),却没有考虑负样本(实际中不应该出现的单词)

对此,我们引入负采样

#### 负采样negative_sampling

1. negative_sampling采用sigmoid函数![image-20230809171057926](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230809171057926.png)
2. negative_sampling引入负样本,也就是噪声,随机选取k的不在该window中的词汇.      对于正向样本,就计算一起出现在window的概率;对于负样本,就计算不一起出现的概率
3. 条件概率的损失函数变为了如下
   1. c是中心词,o是上下文,k是随机选取的不在window中的词![image-20230809165733419](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230809165733419.png)


针对softmax运算导致的每次梯度计算开销过⼤，将softmax函数调整为sigmoid函数，当然对应的含义也由给定中心词，每个词作为背景词的概率，变成了给定中心词，每个词出现在背景窗口中的概率

### co-occurrence matrix(共现矩阵)

前文有提到定义一个window,来表示上下文的范围.我们可以用共现矩阵来计算单词之间共同出现在一个window的次数.

每一行可以当成是一个单词的词向量,并且,有一种直觉,相似的单词会有相似的词向量,比如i like nlp 和 you like nlp两个句子

![image-20230809195606850](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230809195606850.png)

但是,缺点是,如果这样的话,如果词汇量很大,那么一个词向量的维度也会非常多.并且很大可能是一个稀疏矩阵(有很多0)

解决措施,降维,将重要信息集中在维度相对不那么多的密集矩阵中.

解决方法为:**SVD(奇异值分解)**

#### SVD(奇异值分解)

**数据压缩与降维：** 奇异值分解可以帮助我们从大量数据中找到最关键的信息，然后将数据表示为更简洁的形式，从而减少数据的存储空间和处理时间。

a. **原始矩阵：** 首先，我们有一个需要分解的矩阵，可以想象成一个数据表格，其中包含了我们想要分析的数据。

b. **分解：** 我们将这个矩阵分解为三个矩阵的乘积：A = U \* Σ \* V^T^。

- U 是一个正交矩阵，它包含了数据的行之间的关系。
- Σ 是一个对角矩阵，它包含了奇异值，表示了数据的重要程度。
- V^T^ 是 V 的转置，也是一个正交矩阵，它包含了数据的列之间的关系。

### 补充 

1. 刚刚提到,训练过程中,每个单词有两个词向量,其中一个代表中心词,另一个代表上下文.**但是这两个向量的值不是一样的,而是十分相似**.训练的时候才是一个单词对应两个词向量,真正使用的时候的一个单词只对应一个词向量,真正的词向量可能是中心词向量和上下文向量的平均

2. 还有个很神奇的地方,当一个单词有多种含义的时候,也可以只用一个词向量表示.比如star可以表示天文物体的星星,可以表示好莱坞明显,可以表示星星贴纸.  做法就是,将多个star的用法根据不同权重合并为一个向量.听起来很荒唐但是确实有作用![image-20230809212903861](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230809212903861.png)

   ---

   

## 句法结构和依存关系解析

==模型理解句子很重要的一步是理解结构==

### 语法结构

![image-20230810162318536](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230810162318536.png)

**短语结构将单词组织成嵌套成分 **

最基础的起始单元是单词,可分为名字动词形容词副词介词等等

单词组成短语,一整个短语也可以是名词或介词,比如the cuddy cat,    by the door

短语也可以组合成更大的短语

我们还会研究不同种类单词的搭配,比如名词与形容搭配,名词与介词搭配,动词与名词搭配,搭配的单词的数量等等,统称为语法(grammar)

**语法和字典构成一个个句子**

### 依存结构

依存结构展示了一个单词和其他单词的依赖关系,修饰关系

![image-20230810164715492](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230810164715492.png)

比如上面的这个句子,crate是依赖于large,the,in    而look又依赖于crate    kitchen依赖于in the    crate又依赖于kitchen

![image-20230810194104374](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230810194104374.png)

依存结构一般可以表示成树状结构,箭头表示从属关系,箭头旁边的字母表示具体的关系,比如nmod是名词修饰语,nsubj是名义主语,aux是助词

### 歧义问题

既然涉及到结构,那就一定会有一个单词到底修饰哪一个单词,如果修饰不同的部分是否会有不同的意思的问题

比如.  a man kill a woman with knife.     这把刀可以修饰男人,男人用刀杀了女人            这把刀也可以修饰女儿,男人杀了那个带着刀的女人

---



## 句法分析

句法分析（syntactic parsing）是自然语言处理中的关键技术之一，它是对输入的文本句子进行分析以得到句子的句法结构的处理过程。对句法结构进行分析，一方面是语言理解的自身需求，句法分析是语言理解的重要一环，另一方面也为其它自然语言处理任务提供支持

### 依存关系分析

它将句子分析成一颗依存句法**树**，描述出各个词语之间的**依存关系**。

举个例子:"瞧这个可爱的傻瓜"

傻瓜是瞧的对象,所以"傻瓜"依存于"瞧"         可爱的, 小   修饰  傻瓜,所以"可爱的","小"依存于"傻瓜"   其他以此类推

最终我们分析出所有的依存关系

![img](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/v2-ed8fb1f504a56c89a8a458ca01573def_1440w.webp)

在图中我们增加了一个根节点“Root”，这是为了让“瞧”这个字也有依赖的对 象

## 基于转移的依存分析器（Transition-based Parser）

首先,构造三个部分的组合

1. stack(堆栈)
2. buffer(缓存)
3. dependency set(依赖集合)

刚开始stack中只存放"root",需要分析的句子放在buffer中,将buffer中的单词一个个推入stack中,进行分析,分析出来的依存关系结果存放到dependency set中,具体过程如下

一个句子:"i love wuhan"

![img](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/v2-dcbe555bd5e50681a8a6293cad077423_1440w.webp)

知道最后buffer中没有元素,stack中只剩下一个root,结束



### 神经依存分析

我怎么让机器去决定当前的Action呢？即机器怎么知道，Stack中是否构成了依赖关系？机器怎么会知道什么时候使用shift,left arc或者right arc.传统机器学习的方法, 需要做繁重的特征工程.这里的特征，往往 有个二值特征，即无数个指示条件作为特征，来训练模型，可以想象这么高纬度的 特征是十分稀疏的。因此，这种模型的95%左右的解析时间，都花费在计算特征上。这也 是传统方法的最要问题。

神经依存分析靠的是:**「根据当前的==状态==，即Stack、Buffer、Set的当前状态，来构建特征，然后预测出下一步的动作」**

对于单个embedding而言,不仅仅只有词向量本身,还包括了词性和依赖关系,全部嵌入到embedding中.对这三者,都进行低维分布式表示，即通过Embedding的方法，把离散的**word(单词本身)、label(依赖关系标签)、tag(词性)**都转化成低维向量表示。

![img](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/v2-03474659bd9dde15fc4e93c75d9a3e2e_1440w.webp)

![image-20230814204009502](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230814204009502.png)

==神经网络的输入,是**状态**的输入,而不是单词词向量本身的输入了==

![img](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/v2-8cba6c268dee6d93131792cc438a9ad4_1440w.webp)

---



## RNN	

RNN相比word2vec模型好在哪里?

1. 首先,word2vec模型对于**不同的单词有着完全不同的权重**,也就是说,对于输入有着**不对称**的处理,比如下图,就是之间介绍word2vec模型的图,troll2对应的权重是蓝色,is绿色,great橙色,Gymkata粉色.

**![image-20230814213503316](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230814213503316.png)**

2. 对于单词之间,忽略了语序,比如我们word2vec模型,用负采样方法训练的公式举例子.    对于给定的window,只考虑了在给定center单词的情况下,某个outsideword(上下文单词)存在于此window中, 且负采样单词k不在此window的概率.    考虑的是单词之间同时出现的问题,虽然说确实考虑到了单词之间的相似性,但**没有考虑单词的顺序.**![image-20230814214029205](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230814214029205.png)

RNN可以解决以上问题.RNN考虑的时间先后的序列输入,而且对于输入,有着相同的权重参数,对于输入有着对称的处理

### 训练RNN

#### 损失函数

首先,损失函数我们采用**cross-entropy**交叉熵

我们给定一段文本,由多个单词组成,每一个单词输入,都会有一个输出.我们可以每一个时间步的预测单词和下一步的真实单词做交叉熵,算出对应时间步的loss

比如下图,我们利用x~1~预测出下一个单词$\widehat{y1}$,我们就可以跟真实的y~1~计算损失.     输入x~1~和x~2~,就可以预测$\widehat{y2}$,然后跟真实的y2计算损失.     输入x~1~,x~2~,x~3~,就可以预测$\widehat{y3}$,然后跟真实的y3计算损失,最终的损失函数,是前面所有的加起来

![1692022476076](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/1692022476076.png)

#### 反向传播

我们知道,每一个时间步的输出其实也会作为下一个时间步的输入,并且之间连在一起的W~h~矩阵是相同的,那么对于一直重复的W~h~矩阵,该如何计算梯度呢?

把每一个时间步的导数,加起来

![image-20230814222912344](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230814222912344.png)

那对于每一个时间步的导数如何计算呢?    **链式法则**

但是,如果一个句子很长,一路计算导数从终点回到开头非常消耗计算资源,所以对应的解决办法就是设置时间步常数.比如设置为20,就只计算从终点往前20个单词每个单词对应的导数,然后加起来作为梯度,不用一路往前算到起点单词.

**反向传播的过程中可能会出现梯度消失或梯度爆炸的问题**

#### 评估模型

一个模型最终结果的好坏,用什么去衡量呢?    **perplexity**

**perplexity** = 每一个步骤的概率的倒数的累乘,开个1/T次方.(逆概率的几何平均值)              T是单词的个数

评估的时候用的是新的文本,不是训练时的文本,也即是评估用测试集而不是训练集

![image-20230814230311234](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230814230311234.png)

图中上面的公式看起来比较难理解,实则跟图中下方的公式是等价的,跟**交叉熵**十分类似

 

## LSTM

与RNN的不同之处主要是解决了长期记忆的问题

1. LSTM由两个隐藏状态,一个是**cell state**(负责长期记忆),另一个是**hidden state**(负责短期记忆)
2. LSTM可以从cell中**读取,擦除,写入信息**,就好像电脑的内存一样
3. 到底哪些信息应该被读取,擦除,写入,是由**门**来决定的.门是一个向量,门里面的每个元素可以是0--1之间的数
4. **门**是**动态**的,里面的值取决于当前的输入文本

![image-20230815123129929](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230815123129929.png)

**蓝色**部分,遗忘门(forget gate),控制百分之多少的cell state(长期记忆)应该被忘记     ==对应公式f^(t)^==

**绿色**部分吗,输入门(input gate),控制哪一部分新的可能存在的长期记忆应该被写入cell state(真正的长期记忆)中   ==对应公式i^(t)^==

**紫色**部分,输出门(output gate),控制哪一部分的cell(长期记忆),应该写入到hidden state(短期记忆)中    ==对应公式o^(t)^==

**黄色**部分,可能存在的长期记忆(new,cell content)     ==对应公式$\tilde{c^(t)}$==

**上方绿色线**的hadamard乘积和sum,代表**更新cell state(长期记忆)**     ==对应公式c^(t)^==

**红色**部分的蓝色箭头结果和橙色箭头结果hadamard乘积,代表**更新hidden state(隐藏状态)**     ==对应公式h^(t)^==

![image-20230815130503928](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230815130503928.png)

逻辑运算顺序如下

![image-20230815131222985](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230815131222985.png)

### LSTM秘密

LSTM之所以解决了长期记忆的问题,直觉上理解,本质的,关键的一步是**绿色线(cell state)**上**sum**的步骤

![image-20230815131818976](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230815131818976.png)

我们知道,RNN的每一个隐藏状态的输出,要想继续输入到下一个隐藏状态,要**乘上一个权重或权重矩阵**,采用的都是乘法

![image-20230815132047044](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230815132047044.png)

而LSTM多了个==**加法**==的步骤,更像是告诉神经网络,尽管从一个隐藏状态到下一个隐藏状态的需要记住的信息可能不多,但是,请不要忘记她

同时,对于梯度消失和梯度爆炸问题,LSTM有了个加法的引入,也有了很多缓解.因为对于不同的时间步,RNN的矩阵W都会乘在一起,而LSTM的不同的门的矩阵W并不会这样做



## 双向RNN

这里的RNN指的是RNN模型还有其他的一系列变体,不仅仅只是RNN

我们再利用RNN进行文本分析,情绪分析时,我们首先会将一个句子,也即是单词时间序列,输入到RNN模型中,然后,对应每一个时间步(一个隐藏状态),称作该词在本句子中的上下文表述---**上下文表征(语境表示)(contextual representation)**

然后将每一个单词,每一个隐藏状态合在一起,作为句子的编码(sentence encoding),输入到另外的网络进行情感分析

![image-20230815142048364](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230815142048364.png)

但是其中存在的问题是,我们知道,一个单词是的隐藏状态时包含了前面的单词的信息了,比如terribly,结合了前面的the movie was,但是,只结合前面的信息,却没有后面的信息exciting.    现在,这个句子的积极的,"这个电影很刺激".但如果我们没有考虑后面的信息exciting,我们可以会误判,认为这个句子时消极的,"这个电影很糟糕(terribly)"

对应的解决办法就是,创建一个双向RNN,一个正方向传播计算,一个反方向传播计算,然后两个方向的隐藏状态**concatenate**在一起作为句子编码(sentence encoding)

![image-20230815142859840](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230815142859840.png)

正方向传播计算的RNN和反方向计算的RNN有着**不同的网络权重**

![image-20230815143033238](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230815143033238.png)

​    

## 神经机器翻译(neural machine translation)

### seq2seq模型

seq2seq模型分为两个部分,encoder(编码器)和decoder(解码器)两个部分,每个部分就是一个LSTM模型

按照LSTM模型的方式,将原句子按照序列形式输入到LSTM模型中,最终的隐藏状态将作为decoder的初始状态输入

先把<START>输入decoder,作为翻译后的句子的开头,该隐藏状态会生成一个单词he,然后一个隐藏状态的输出将作为下一个状态的输入,一直循环,知道翻译出<END>

![image-20230815211307181](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230815211307181.png)

这也可以理解为一个条件语言模型,说他是语言模型是因为他做的就是不断预测下一个单词的概率,条件是因为所有的预测都是基于原来的句子计算的

![image-20230815223937375](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230815223937375.png)

目标句子的下一个单词的概率是基于目标句子前文的单词还有原句子而计算的



#### 如何训练seq2seq

1. 首先,我们要准备好语料库,里面包含原句子和平行句子(也就是正确翻译后的句子)
2. 将原句子序列输入encoder,将**正确**的句子序列输入decoder
3. decoder的每一个状态都会产生一个预测$\widehat{y}$,我们利用y和$\widehat{y}$计算损失函数.   最终的损失函数就是decoder的所有状态的损失加起来
4. 计算梯度反向传播,既可以更新decoder的参数,也可以更新encoder的参数

![image-20230815221249579](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230815221249579.png)

### beam search decoding

我们知道,在decode的时候,每一个状态的最大概率的那个单词将会作为下一个状态的输入.但是,问题是,第一个预测的单词概率最大(最优),不代表后面接连预测的单词也是最优.   就好比第一个单词概率0.99,但是后面的单词却预测得很差,有种贪图眼前便宜的意思.这种方法就是**贪心decode**

而**beam decode**,贪心策略一个改进。思路也很简单，就是稍微放宽一些考察的范围。在每一个时间步，不再只保留当前分数最高的**1**个输出，而是保留**num_beams**个。当num_beams=1时集束搜索就退化成了贪心搜索。

下图是一个实际的例子，每个时间步有ABCDE共5种可能的输出，即，图中的num_beams=2，也就是说每个时间步都会保留到当前步为止条件概率最优的2个序列。

![img](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/v2-a760198d6b851fc38c8d21830d1f27c9_1440w.webp)

- 在第一个时间步，A和C是最优的两个，因此得到了两个结果`[A],[C]`，其他三个就被抛弃了；
- 第二步会基于这两个结果继续进行生成,在A这个分支可以得到5个候选人,C也同理得到5个,   此时会对这10个进行统一排名，再保留最优的两个，即图中的`[AB]`和`[CE]`；
- 第三步同理，也会从新的10个候选人里再保留最好的两个，最后得到了`[ABD],[CED]`两个结果。

这种方法不能保证一定生成的句子是全局最优,但可以**一定程度上避免局部最优**.

可以发现，beam search是一种牺牲时间换性能的方法



## 多层RNN(multi-layer RNN)(stacked RNNS)

对于一些比较简单的,浅层的分析,比如只是对一个句子做一些简单的分析,浅层RNN可以,但是当追求更好的效果时候,比如考虑句子的情绪,句子的语境等等更多的特征,更复杂的特征,就需要引入多层RNN

![image-20230815223145802](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230815223145802.png)

该层某个状态的输出,不仅会通往该层的下一个状态,还会通往该状态的下一层

一般是2--4层,2层比1层效果好了很多,而3层4层的改进其实不多,甚至变坏,到底选用多少层很大程度上取决于我们的数据



## Attention

==attention的核心是:对于decoder(解码器)的每一步,**建立一个跟encoder(编码器)的直接联系**,这个联系可以让decoder专注于原句子的某个部分==

其实这跟我们人工翻译一个句子一样,我们看了一眼原句子,虽然脑子里确实会记住个大概,然后在翻译的时候会结合记住的原句子信息去翻译.    但是我们翻译的过程中肯定会**回头看几眼原句子**,继续翻译,然后再回过头再看几眼原句子,再继续翻译.       而这个**回头看**的动作,好比attention.

### 具体步骤

1. 首先,对于decoder的某一个步骤,我们跟encoder的所有隐藏状态计算相似性,可以是点乘,也可以是余弦相似性
2. 因此,对于encoder的每一个隐藏状态,都有了一个相似性分数(注意力分数)(score)
3. 用softmax函数,将score转化为注意力概率分布(attention distribution)
4. 利用这个attention distribution,对encoder的每一个隐藏状态进行**加权和**,得到了注意力输出(attention output)
   1. 这个attention output 主要包含的是高注意力的encoder的隐藏状态的信息
5. 将这个attention output跟decoder的隐藏状态结合(concatenate),利用softmax函数计算最终预测单词输出的概率分布
   1. 如果不加attention,就是直接对decoder的隐藏状态进行softmax得到最终预测单词的概率分布

<img src="https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230816115959296.png"/>



## self attention

之前的笔记已经介绍过self attention的大部分关键知识,这里只做补充

我们来谈谈self attention存在的问题,以及如何解决对应的问题

1. self attention将所有输入的词向量的Q,K,V乘来乘去,同时输入,同时输出,并行化处理,却没有考虑词序,我们知道,对于一个句子来说,单词的顺序是十分重要的.
   1. 对应的解决办法是:为每一个词向量添加**位置编码**
      1. 正弦位置编码

![image-20230816165117814](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230816165117814.png)



2. 深度学习很重要的其中一个因素是,引入了**非线性**模块,而self attention中的Q,K,V之间的相乘是一种线性变换.即使我们叠加多个self attention层,得到的效果也只是平均了最终的V向量.
   1. 对应的解决办法是:引入非线性模块
      1. 对于输出,增加一个**前馈层**

![image-20230816165519932](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230816165519932.png)



3. 我们再分析一个句子的时候,对于某一个单词,确实应该既要考虑过去的单词,也要考虑未来的单词.所以,我们有双向RNN网络.但是,我们在**训练**语言模型的时候,我们做到是不断预测接下来的单词,也就是,未来的单词我们不应该知道.所以,相应的Q,K,V矩阵的应该随着时间序列做出调整
   1. 对应的解决办法是,掩盖未来的self attention
      1. masked self attention.我们根据不同的时间步,改变我们的K和Q.       我们的注意力分数是Q^T^K,对于过去的单词,注意力分数就是Q^T^K,,   而对于未来的单词,注意力分数为－∞

![image-20230816170819598](https://spasmodic.oss-cn-hangzhou.aliyuncs.com/image-20230816170819598.png)