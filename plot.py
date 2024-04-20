import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def entropy(p):
    return p * np.log2(p) + (1 - p) * np.log2(1 - p)


def binary_cross_entropy(x1, x2):
    return entropy(x1) + entropy(x2) + entropy(2 - 3*x1)


def square_relu(x):
    return np.maximum(0, x) ** 2


def centre(x1, x2):
    return square_relu(x1) + square_relu(x2) + square_relu(2 - x1 - x2)


def distance(x1, x2, w1, w2, b):
    return np.abs(w1 * x1 + w2 * x2 + b) / np.sqrt(w1 ** 2 + w2 ** 2)


def custom_loss(x1, x2):
    return distance(x1, x2, 1, 0, 0)**2 + distance(x1, x2, -1.732, 1, -1.732)**2 + distance(x1, x2, 1.732, 1, -1.732)**2


# 创建一个 figure 和一个 3d 绘图区域
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 创建 p1 和 p2 的网格数据
p1 = np.linspace(-2, 2, 100)  # 避免 log(0)
p2 = np.linspace(-2, 2, 100)
P1, P2 = np.meshgrid(p1, p2)


Cross_Entropy = custom_loss(P1, P2)
# 绘制图像
surf = ax.plot_surface(P1, P2, -Cross_Entropy, cmap='viridis')

# 添加坐标轴标签
ax.set_xlabel('p1')
ax.set_ylabel('p2')
ax.set_zlabel('Cross-Entropy Loss')

# 添加 color bar
fig.colorbar(surf, shrink=0.6, aspect=5)

# 显示图像
plt.show()
