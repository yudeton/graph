import pandas as pd
import matplotlib.pyplot as plt

def load_and_process_csv(file_path):
    """
    读取和处理 CSV 文件，返回处理后的数据。
    """
    data = pd.read_csv(file_path)
    # data = data[data['Value'] <= 1000]  # 清除 'Value' 列中大于 2000 的数据点
    # data['Value'] = data['Value'] + 200  # 将 'Value' 列中的所有值增加 200
    # data = data[data['Step'] <= 10000]  # 只显示 'Step' 列中不超过 15000 的数据点
    return data['Value']
def load_and_process_csv2(file_path):
    """
    读取和处理 CSV 文件，返回处理后的数据。
    """
    data = pd.read_csv(file_path)
    # data = data[data['Value'] <= 2000]  # 清除 'Value' 列中大于 2000 的数据点
    # data['Value'] = data['Value'] + 200  # 将 'Value' 列中的所有值增加 200
    data = data[data['Step'] <= 10000]  # 只显示 'Step' 列中不超过 15000 的数据点
    return data['Value']
# 读取和处理四个 CSV 文件
data1 = load_and_process_csv('DQN/DQN_episode_return.csv')
data2 = load_and_process_csv('DQN+/run-.-tag-trained-model_Episode_Return_ (1).csv')
data3 = load_and_process_csv2('PDQN/run-.-tag-Episode_Return (1).csv')
data4 = load_and_process_csv2('PDQN+/run-.-tag-Episode_Return.csv')

# 将数据放在一个列表中，以便绘制箱型图
data_to_plot = [data1, data2, data3, data4]

# 创建箱型图
plt.figure(figsize=(10, 6))
plt.boxplot(data_to_plot, vert=True, patch_artist=True, labels=['DQN', 'DQN+', 'P-DQN', 'P-DQN+'])
plt.xlabel('DRL')
plt.ylabel('Rewards')
plt.title('Episode Rewards Box Plot')
plt.show()
