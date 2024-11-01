import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def f(x):
    return x*x                      #在此处输入 f(x) 的表达式          
# 创建输入数据
x = np.linspace(-10, 10, 400)  # 在 -10 到 10 之间
# 计算函数值
y = f(x)

# 绘制图形
plt.plot(x, y, label='f(x) = exp(-sqrt(x)/0.2')

# 添加标题和标签
plt.title('Function Plot of f(x) =  ')                  #手动输入添加标签，不重要
plt.xlabel('x')
plt.ylabel('f(x)')
# 显示网格
plt.grid(True)

# 添加图例
plt.legend()

# 显示图形
plt.show()