import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import List

def smooth(scalars: List[float], weight: float) -> List[float]:
    """
    Apply Exponential Moving Average (EMA) smoothing to a list of scalar values.

    The implementation follows the method used in TensorBoard:
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699

    Args:
        scalars (list of float): The list of scalar values to smooth.
        weight (float): The smoothing weight. A higher weight means more smoothing.

    Returns:
        list of float: The smoothed values.
    """
    last = 0
    smoothed = []
    num_acc = 0

    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1

        # De-bias the result
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight

        smoothed.append(smoothed_val)

    return smoothed

# 假设 'data' 是包含你的训练数据的 DataFrame
data = pd.read_csv('DQN+/run-.-tag-trained-model_loss_log_ (1).csv')

# 清除 'Value' 列中大于 2000 的数据点
# data = data[data['Value'] <= 100000]
# data['Value'] = data['Value'] + 400
# 只显示 'Step' 列中不超过 15000 的数据点
# data = data[data['Step'] <= 10000]
# 设置平滑参数（例如，0.6）
alpha = 0.99

# 计算指数移动平均线
data['Smoothed'] = smooth(data['Value'].tolist(), alpha)

# 绘制原始数据和平滑数据
plt.figure(figsize=(12, 6))
plt.plot(data['Step'], data['Value'], label='Original', alpha=0.5)
plt.plot(data['Step'], data['Smoothed'], label='Smoothed', linestyle='--')
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('DQN+ Loss')
plt.legend()
# plt.xlim(0, 15000)
plt.show()

# # 绘制原始数据和平滑数据
# plt.figure(figsize=(12, 6))
# plt.plot(data['Step'], data['Value'], label='Original', alpha=0.5)
# plt.plot(data['Step'], data['Smoothed'], label='Smoothed', linestyle='--')
# plt.xlabel('Step')
# plt.ylabel('Value')
# plt.title('DQN+ Reward')
# plt.legend()
# plt.show()
