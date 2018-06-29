#Logisstic Regression as Neural Network
@(deeplearning.ai)
##RGB
![](https://ws2.sinaimg.cn/large/006tNc79gy1fsmnjmltqsj31iy0v2np7.jpg)
- 原始图片是$64\times64$
- R、G、B分别是$64\times64$
- 所有原始图片$X$向量有$64\times64\times64\times3=12288$维，即输入向量有12288维。$n_x=12288$
## 表示方法
![](https://ws3.sinaimg.cn/large/006tNc79gy1fsmojysd9nj31is0uo4d6.jpg)
- 事实证明在神经网络中，每个x为列向量，运算时会相对简单。
  `这里注意：机器学习中大多数使用的是叉掉的表示方法，即每个x为行向量`
- 运行X.shape，得到的是$(n_x,m)$
## Logistic Regression
![](https://ws4.sinaimg.cn/large/006tNc79gy1fsmp8zq7yxj31j40v6aq6.jpg)
- 需要得到一个概率（0到1之间），而不是回归的值（可能是负数也可能远大于1），此时用到sigmoid函数
  $$\sigma(z)=\frac{1}{1+e^{-z}}$$
- 通常会将参数w和参数b分开看待，所以这门课中`不会`使用红色笔的表示方法
## Cost Function
![](https://ws4.sinaimg.cn/large/006tNc79gy1fsmpt8h53mj31je0v41du.jpg)
- 为了优化$w$和$b$，我们提出了代价函数
- 你可能曾经这样定义损失函数：$$Loss(\hat{y},y)=\frac{1}{2}(\hat{y}-y)^2$$
  但是`逻辑回归`里一般不这样定义，因为当研究参数时，我们讨论的优化问题将会变成`非凸问题`，所以会产生多个局部最优解，梯度下降法就无法找到全局最优解
- 逻辑回归中的损失函数：
  $$Loss(\hat{y},y)=-(ylog\hat{y}+(1-y)log(1-\hat{y}))$$
  即：（图中绿色推导）
  y=1时，要使损失函数小，$\hat{y}$要尽可能大
  y=0时，要使损失函数小，$\hat{y}$要尽可能小
- 整体损失函数`Cost Function` J
  m个单一样本的损失函数`Loss Function`L取平均，展开后就是关于$w,b$的函数
## Gradient Descent
![](https://ws2.sinaimg.cn/large/006tNc79gy1fsmrjde0h0j31kw0vre2v.jpg)
- 代价函数J是一个凸函数（`convex function`）
- 使用一些初始值来初始化$w,b$，通常用0或者随机数来初始化
- 朝最陡的下坡方向走一步，尽可能快的下坡
  ![](https://ws4.sinaimg.cn/large/006tNc79gy1fsmrp4qgyqj31kw0vptrj.jpg)
- $w(新) = w - \alpha\frac{dJ(w)}{dw}$
- $\alpha$表示学习率：控制每一次梯度下降步长的大小
- $\frac{dJ(w)}{dw}$是导数，即当前w的斜率，在代码中简写为dw，整个更新过程简写为：$$w:=w-\alpha dw$$
- $\frac{dJ(w)}{dw}$若为正数，w往左移：减小w，反之亦然。
- 无论初始化时w在左边还是在右边，w都会朝着局部最优的方向移动
- 数学知识：如果J只含一个参数，用$d$（导数）；J含两个参数即为偏导数，用$\delta$（偏导）
- 偏导：函数`关于其中某一个变量`在对应点处的斜率
##Derivatives
![](https://ws2.sinaimg.cn/large/006tNc79gy1fsms8z8nwmj31ig0um182.jpg)
##Logistic Regression Derivatives (逻辑回归中的导数：反向传播)(`单个样例`)
![](https://ws4.sinaimg.cn/large/006tKfTcgy1fsnhtaxm6xj31kw0wctp6.jpg)
- 假设有2个特征$x_1,x_2$
- 在逻辑回归中我们要做的就是：修改$w_1,w_2,b$去减小$L(a,y)$
  ![](https://ws2.sinaimg.cn/large/006tKfTcgy1fsnhpeqdt6j31j00v0arg.jpg)
- 计算结果
  $dz=a-y$
  $dw_1=x_1dz$
  $dw_2=x_2dz$
  $db=dz$
- 更新参数
  $w_1:=w_1-\alpha dw_1$
  $w_2:=w_2-\alpha dw_2$
  $b:=b-\alpha db$
## Logistic Regression Derivatives(`m个样例`)
![](https://ws3.sinaimg.cn/large/006tKfTcgy1fsni3rgdg5j31kw0vtari.jpg)
- m个样本时对应`Cost Function`的是$$J(w,b)=\frac{1}{m}\sum_{i=1}^mL(a^{(i)},y^{(i)})$$：
### `一次梯度下降`具体算法过程如下：
![](https://ws3.sinaimg.cn/large/006tKfTcgy1fsnispa0mlj31kw0wanhz.jpg)
- 一次梯度下降包括2次for循环
- 1.是m个样本
- 2.是每个样本的维数
- 使用这种显示的for loop会使算法不够高效
- `Vectorization`会帮你摆脱这种显示的for loop，我需要一个for循环也不用！