import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取csv文件
df = pd.read_csv('height_data.csv', header=0)
data = df['height'].values.ravel()

# 设置模型参数


# 初始化模型参数
weights = np.array([0.5, 0.5])
means = np.array([[165.0], [178.0]])
standard_deviation = np.array([[[5.0]], [[5.0]]])

n_components = 2
n_features = 1
n_samples = len(data)
max_iter = 1000
tolerance = 1e-6


def gaussian_pdf(data, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * (np.exp(-0.5 * (data - mu) ** 2 / sigma ** 2))


# EM算法的E步和M步
for i in range(max_iter):
    # E步
    likelihood = np.zeros((n_samples, n_components))
    for k in range(n_components):
        likelihood[:, k] = weights[k] * gaussian_pdf(data, means[k], standard_deviation[k])
    posterior = likelihood / np.sum(likelihood, axis=1, keepdims=True)

    # M步
    weights = np.mean(posterior, axis=0)
    means = np.sum(data.reshape(-1, 1) * posterior, axis=0) / \
        np.sum(posterior, axis=0)
    standard_deviation = np.sqrt(np.sum((data.reshape(-1, 1) - means)**2 *
                         posterior, axis=0) / np.sum(posterior, axis=0))

    # 计算对数似然函数值，检查收敛性
    log_likelihood = np.sum(np.log(np.sum(likelihood, axis=1)))
    if i > 0 and np.abs(log_likelihood - prev_log_likelihood) < tolerance:
        break
    prev_log_likelihood = log_likelihood

# 输出模型参数
print('weights:', weights)
print('means:', means)
print('standard deviation:', standard_deviation)

x = np.linspace(150, 195, 200)
y = weights[0] * gaussian_pdf(x, means[0], standard_deviation[0]) + weights[1] * gaussian_pdf(x, means[1], standard_deviation[1])
# 绘制数据的直方图
plt.figure()
plt.hist(data, bins=35, rwidth=0.9, label='Sample data')
plt.plot(x, y * 2000, 'g-', linewidth=2, label='Fitted curve')
plt.xlabel('Height (cm)')
plt.ylabel('Count')
plt.title('Distribution of Heights')
plt.legend()
plt.show()
