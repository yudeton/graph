import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# 读取 CSV 文件
data = pd.read_csv('DQN_loss.csv')

# 查看数据结构
print(data.head())

# # 绘制图表
# plt.figure(figsize=(10, 6))
# plt.plot(data['Step'], data['Value'])
# plt.xlabel('Step')
# plt.ylabel('Reward')
# plt.title('DQN Episode Reward')
# plt.show()

# # 绘制图表
# plt.figure(figsize=(10, 6))
# plt.plot(data['Step'], data['Value'])
# plt.xlabel('Step')
# plt.ylabel('Loss Value')
# plt.title('DQN Loss')
# plt.show()

# 设置平滑参数（例如，0.6）
alpha = 0.05

# 计算指数移动平均线
data['Smoothed'] = data['Value'].ewm(alpha=alpha, adjust=False).mean()

# 绘制原始数据和平滑数据
plt.figure(figsize=(12, 6))
# plt.plot(data['Step'], data['Value'], label='Original', alpha=0.5)
plt.plot(data['Step'], data['Smoothed'], label='Smoothed', linestyle='--')
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Training Progress with Exponential Moving Average Smoothing')
plt.legend()
plt.show()

# # 计算移动平均线
# data['Moving_Avg'] = data['Value'].rolling(window=10).mean()

# # 绘制原始数据和移动平均线
# plt.figure(figsize=(10, 6))
# plt.plot(data['Step'], data['Value'], label='Original')
# plt.plot(data['Step'], data['Moving_Avg'], label='Moving Average', linestyle='--')
# plt.xlabel('Step')
# plt.ylabel('Value')
# plt.title('Training Progress with Moving Average')
# plt.legend()
# plt.show()