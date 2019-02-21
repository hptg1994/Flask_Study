import numpy as np
import matplotlib.pyplot as plt

# 这里相当于是随机X维度X1，rand是随机均匀分布
# random.rand 这里是生成二维函数100行1列(从0到1之间选值)
X = 2 * np.random.rand(100, 1)

# 人为的设置真实的Y一列，np.random.randn(100, 1)是设置error，randn是标准正太分布，所以取0的附近可能性比较大
y = 4 + 3 * X + np.random.randn(100, 1)
print(y)

 # 整合X0和X1（c_:combine两个向量，X0和X1，现在就是两列100行了）
X_b = np.c_[np.ones((100, 1)), X]
print(X_b)

# 常规等式求解theta
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print(theta_best)

# 创建测试集里面的X1
X_new = np.array([[0], [2]]) 
X_new_b = np.c_[(np.ones((2, 1))), X_new]
# print(X_new_b)
y_predict = X_new_b.dot(theta_best)
# print(y_predict)

plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()
