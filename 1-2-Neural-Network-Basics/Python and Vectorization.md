# Python and Vectorization

@(deeplearning.ai)

## 什么是向量化？
![](https://ws4.sinaimg.cn/large/006tKfTcgy1fsnjkveuerj31j60uudt9.jpg)
- 左边是非向量化实现（`非常的慢`）
- 右边是向量化实现Python：
```python
import numpy as np
Z=np.dot(w,x)+b
```
其中`np.dot(w,x)`就是$w^Tx$
- GPU和CPU都有`并行化`的指令：叫`SIMD`sinle instruction multi data
- Python中的numpy就是利用`并行化`去更快的计算
- GPU的并行计算被认为更快，但事实上CPU也不会太差
- 无论什么时候：避免使用for loop，尽量使用内置函数(built-in function)
  ![](https://ws1.sinaimg.cn/large/006tKfTcgy1fsnkh4vml0j31j40us7k0.jpg)
- 左边是非向量化实现，先初始化一个n维的0向量列。再for遍历求$e^{v_i}$
- 直接`u.exp(v)`
- Numpy中其他支持向量值的函数：
```python
# 底数是2
x = np.array([0, 1, 2, 2**4])
np.log2(x)
array([-Inf,   0.,   1.,   4.])

# 底数是e
np.log([1, np.e, np.e**2, 0])
array([  0.,   1.,   2., -Inf])

# 比较大小，返回大的
np.maximum([2, 3, 4], [1, 5, 2])
array([2, 5, 4])
```
## 将向量化应用到逻辑回归中
![](https://ws3.sinaimg.cn/large/006tKfTcgy1fsnr8plkmij31de0raasw.jpg)
- 非向量化时，我们有2个for loop
- 先试着删掉1个for循环
- 不再把$d_w初始化为0$，而是`dw=np.zeros(n_x,1)`
- 保留了对m个样本的循环
## 通过向量化计算预测
![](https://ws1.sinaimg.cn/large/006tKfTcgy1fsnrszdcy2j31jc0uqkc7.jpg)
- Python中的广播(broadcasting)
- 当b是一个实数（1x1向量），但是把这个实数b加到向量中时，python会自动把b拓展成一个1xm的行向量，这叫做Python的广播
- 得到A（1xm）的向量，即为预测向量。
## 通过向量化计算反向传播
![](https://ws4.sinaimg.cn/large/006tKfTcgy1fsnu87wxzoj31kw0ux7sc.jpg)
## 对比
![](https://ws2.sinaimg.cn/large/006tKfTcgy1fsnuwkcbxuj31kw0vcav6.jpg)
- 迭代1000次
## 继续说向量化
![](https://ws4.sinaimg.cn/large/006tKfTcgy1fsojipbmggj30vy0q4423.jpg)
- axis=0 表示沿着垂直方向
- 如果不确定矩阵的维数，可以调用`reshape`
## 继续说Python广播
![](https://ws3.sinaimg.cn/large/006tKfTcgy1fsojn28wvaj31je0ugamo.jpg)
## numpy vectorz 中的BUG
![](https://ws4.sinaimg.cn/large/006tKfTcgy1fsomk5vmbej310q0rojuj.jpg)
- Andrew Ng建议：在神经网络中不要使用形如(n, )的向量
- 而是使用明确的向量
  ![](https://ws2.sinaimg.cn/large/006tKfTcgy1fsonfvqg63j310g0ngn0o.jpg)
- 注意：原理的向量是一个`秩为1的`一维数组，现在的向量是一个`矩阵（有2个方括号）`
- 使用下列的用法：
  ![](https://ws3.sinaimg.cn/large/006tKfTcgy1fsonc1nrt4j31i80t0add.jpg)
## 逻辑回归中的损失函数是怎么来的？
![](https://ws1.sinaimg.cn/large/006tKfTcgy1fsoox9gdexj31gy0t2nh1.jpg)
- 最小化损失函数===>最大化对数（因为对数单调递增）===>最大化概率
## 作业错题
- 广播的例子
  ![](https://ws3.sinaimg.cn/large/006tKfTcgy1fsopr8zvacj30r60g840d.jpg)
  ![](https://ws3.sinaimg.cn/large/006tKfTcgy1fsoptvbd9rj315m0xsq7r.jpg)
- 在numpy中`*`运算符将会形成`element-wise multiplication`，它与`np.dot`不同，如果此处的`b.size`是`(4,1)or(1,3)`，则Python将会产生`broadcasting`
  ![](https://ws1.sinaimg.cn/large/006tKfTcgy1fsoq9yikhlj312s0c6jtb.jpg)
  ![](https://ws3.sinaimg.cn/large/006tKfTcgy1fsoqavepptj30xm08yta1.jpg)
## numpy中的乘法
- np.dot 乘法
- np.outer 外积
- np.multiply 点积
  ![](https://ws2.sinaimg.cn/large/006tKfTcgy1fsowd1nbnsj31iw0p4jy5.jpg)
  ![](https://ws4.sinaimg.cn/large/006tKfTcgy1fsowdp4c6oj310w0l4gp6.jpg)