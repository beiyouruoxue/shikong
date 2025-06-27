import numpy as np
import math

# 加载数据集（请替换实际路径）
data = np.load('nyc_taxi.npy')  # 形状应为(17520, 75, 75, 1)

# 基本参数设置
T = data.shape[0]  # 时间点数量 (17520)
n_nodes = data.shape[1]  # 节点数量 (75)
h = 96  # 历史窗口大小（24小时*4=96，即使用前一天数据）

# 初始化预测结果数组
prediction = np.zeros((T, n_nodes, n_nodes))
actual = np.zeros((T, n_nodes, n_nodes))

# 移除多余的维度 (17520, 75, 75, 1) -> (17520, 75, 75)
if data.ndim == 4 and data.shape[3] == 1:
    data = data.squeeze(axis=-1)

# 生成预测（从第h个时间点开始）
for t in range(h, T):
    # 获取历史窗口数据（t-h 到 t-1）
    history_window = data[t - h:t, :, :]

    # 计算历史平均值作为预测
    prediction[t] = np.mean(history_window, axis=0)

    # 存储实际值
    actual[t] = data[t]

# 向量化优化版本（内存效率更高）
valid_pred = np.zeros((T-h, n_nodes, n_nodes))
valid_actual = data[h:]

for i, t in enumerate(range(h, T)):
    valid_pred[i] = np.mean(data[t-h:t], axis=0)

mae = np.mean(np.abs(valid_pred - valid_actual))
rmse = np.sqrt(np.mean((valid_pred - valid_actual)**2))

print(f"历史窗口大小 h = {h} (24小时)")
print(f"有效预测时间点数量: {len(valid_pred)}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# 可选：保存预测结果
# np.save('ha_predictions.npy', prediction)
# np.save('actual_values.npy', actual)