# [第三週]Python強大的秘密：numpy、向量化與各式套件

### 正規讀套件的方法(一)
讀套件裡面函數的方法

    form 套件 import 函數
    
![](https://i.imgur.com/IrqIcae.png)

<div class="title">引入 random 函數</div>

```python
from random import *

for i in range(1,50):
    print(randint(1,10), end=", ")
```


雖然指令一模一樣，但功能不同： `random` 會印出 10，`numpy.random` 不會印出 10。這樣的差異會造成困擾，因此不推薦引入時使用 `import *` 的方式。

![](https://i.imgur.com/YYpKiwk.png)


---
### 正規讀套件的方法(二）
每次下指令都要打 numpy 相當不便，因此可以透過以下指令來指派一個綽號給它（幾乎所有高手都以 np 取代 numpy，因此在這也會用 np 取代）
    
    import numpy as np

![](https://i.imgur.com/1YC5aw4.png)

---
### 數據分析的標準動作
若要做數據分析，第一部標準動作:
讓 jupyter notebook 能夠出現 matplotlib 的圖片
 
```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
```

```python
plt.plot([-2,3,-1,5,7])
```
![](https://i.imgur.com/dndzckv.png)


```python
plt.plot(np.random.randn(100))
```
![](https://i.imgur.com/e1YvOoz.png)


---
### 處理一整個 List 的數字
若要將資料塞進陣列裡，可使用 `.append` 的方式
```python
price = [2567.7, 540.2, 899.9]
m = 30.5
result = [ ]
for k in price:
    twd = k * m
    result.append(twd)
```

![](https://i.imgur.com/dsr0jaH.png)

---
### Array 真是太炫了
數據分析的基本概念：能不用迴圈就盡量避免使用，會導致效能不佳。

array 可以做類似向量的運算，尚未把數列轉換成 array 之前，無法將數列與一個數字相乘，透過 `np.array` 將數組轉換成陣列後便可以進行相乘。

```python
price = [2567.7, 540.2, 899.9]
m = 30.5
p = np.array(price)
result = p * m
```

![](https://i.imgur.com/NFCAYmP.png)

---

### 用 Array 算成績

<div class="title">.sum()函數</div>
透過 `.sum()` 可以將陣列內數值相加

```python
grade = np.array([85, 70, 82])
weights = np.array([0.3, 0.4, 0.3])
g = grade * weights
g.sum()
```

![](https://i.imgur.com/UKiARg5.png)


<div class="title">np.dot()函數</div>

將 array1 和 array2 做內積，對於 index 為 1 的數組，執行對應位置相乘，然後再相加。

    np.dot(grade, weights)


<div class="title">array 大變身</div>

如果一個陣列是 10 維度（有十個數字），若要將它轉換成 2x5 的矩陣該怎麼做？這是在數據分析裡非常重要的動作。

    A = np.random.randn(100)


![](https://i.imgur.com/8ZY46Nd.png)

以上這 100 個亂數的平均值是 0，標準差是 1

若要讓它的平均為 50，標準差為 10，該怎麼做？

    A * 10 + 50

#### 更改 Array 的形狀

現在 A 是 100 維的狀態，如果要將其轉換成 5x20 維度的資料要怎麼做？
我們可以先透過 `array.shape` 看 array 現在的形狀
 
 ![](https://i.imgur.com/5S1OH7I.png)
 
 使用 
 
    array.shape = (新的形狀)
     
所以透過

    A.shape = (5,20)
    
![](https://i.imgur.com/a9kOzoU.png)

#### reshape() 函數
reshape 是一個函數，所以直接在後面括弧中寫進要改變的形狀。但 reshape 是產生一個新的 array，他不會改變原本的 array

    A.reshape(100,1)

![](https://i.imgur.com/0nbTx3v.png)


---
### 快速生成 Array

快速生成一個特並範圍 array 的方法：
```python
xy = [[x,y] for x in range(10) for y in range(5)]
```

將生成的數列轉為 array
```python
xy = np.array(xy)
```

要產出十個 0 的 array：

```python
np.zeros(10)
```
![](https://i.imgur.com/Vii5Wqz.png)

生成一個值都是零的array，也可以產出矩陣形式：
np.zeros(你要的形狀)

```python
np.zeros((3,4))
```

![](https://i.imgur.com/MIUOcuw.png)

生成都是 1 的 array
```python
np.ones(5)
```

:::warning
生成一個n乘n的單位矩陣：

```python
np.eye(n)
```
![](https://i.imgur.com/ZOKDmiZ.png)

生成一個值都是1的單位矩陣：

```python
np.ones(你要的形狀)
```
:::
















<style>
.title{
    font-size: 18px;
    font-weight: 700;
    border-left:4px solid #F89C2D;
    background-color: #FEF7ED;
    margin: 5px 5px 5px 0px;
    padding: 5px;
}
</style>