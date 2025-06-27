import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt


# 假设数据加载函数 - 实际使用时替换为真实数据加载
def load_data():
    """模拟加载纽约出租车OD矩阵数据"""
    # 真实数据应替换为: np.load('nyc_taxi_od.npy')
    # return np.random.rand(17520, 75, 75, 1)  # 随机数据用于演示
    return np.load("nyc_taxi.npy")

# 1. 加载原始数据
print("Loading data...")
data = load_data()  # 形状: (17520, 75, 75, 1)
print(f"原始数据维度: {data.shape}")


# 2. 数据预处理
def preprocess_data(data):
    """数据预处理和特征工程"""
    # 转换为2D: [时间, 空间特征]
    spatial_data = data.reshape(data.shape[0], -1)  # 形状: (17520, 5625)

    # 提取时间特征
    hours = np.arange(data.shape[0]) % 24
    weekdays = (np.arange(data.shape[0]) // 24) % 7

    # 周期性编码时间特征
    time_features = np.column_stack([
        np.sin(2 * np.pi * hours / 24),
        np.cos(2 * np.pi * hours / 24),
        np.sin(2 * np.pi * weekdays / 7),
        np.cos(2 * np.pi * weekdays / 7)
    ])

    # 空间特征降维 (保留95%方差)
    pca = PCA(n_components=0.95)
    spatial_reduced = pca.fit_transform(spatial_data)
    print(f"空间特征降维: {spatial_data.shape[1]} -> {spatial_reduced.shape[1]}")

    # 合并时空特征
    all_features = np.hstack([spatial_reduced, time_features])

    return all_features, spatial_data, pca


print("Preprocessing data...")
X, spatial_data, pca = preprocess_data(data)

# 3. 构建预测目标 (下一时刻的OD矩阵)
y = np.roll(spatial_data, -1, axis=0)[:-1]  # 下一个时间步作为目标
X = X[:-1]  # 对齐时间步
print(f"特征维度: {X.shape}, 目标维度: {y.shape}")

# 4. 划分训练测试集 (按时间顺序)
test_size = int(0.2 * X.shape[0])
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# 5. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 6. 定义多输出SVR封装器
class MultiOutputSVR:
    """多输出SVR封装器，处理多目标预测"""

    def __init__(self, ** kwargs):
        self.models = []
        self.params = kwargs

    def fit(self, X, y):
        """训练多个SVR模型 (每个OD点一个模型)"""
        print(f"训练 {y.shape[1]} 个SVR模型...")
        self.models = []

        # 进度跟踪
        for i in range(y.shape[1]):
            model = SVR(**self.params)
            model.fit(X, y[:, i])
            self.models.append(model)

            # 打印进度
            if (i + 1) % 10 == 0 or (i + 1) == y.shape[1]:
                print(f"已训练 {i + 1}/{y.shape[1]} 个模型")
        return self

    def predict(self, X):
        """生成预测结果"""
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        return predictions


# 7. 训练模型 (使用小样本演示，实际应使用更多超参数)
print("\nTraining model...")
start_time = time.time()

# 设置初始超参数 - 实际应使用网格搜索优化
params = {
    'kernel': 'rbf',
    'C': 10,
    'epsilon': 0.1,
    'gamma': 'scale'
}

# 为加速演示，使用部分OD点 (实际应使用全部)
DEMO_MODE = True  # 设为False使用完整OD矩阵
if DEMO_MODE:
    print("DEMO模式: 仅预测前500个OD点")
    y_train = y_train[:, :500]
    y_test = y_test[:, :500]

# 初始化并训练模型
multi_svr = MultiOutputSVR(**params)
multi_svr.fit(X_train_scaled, y_train)

print(f"训练完成! 耗时: {time.time() - start_time:.2f}秒")

# 8. 预测测试集
print("Predicting...")
y_pred = multi_svr.predict(X_test_scaled)


# 9. 评估模型
def evaluate_predictions(y_true, y_pred):
    """评估预测性能"""
    # 总体指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"整体MAE: {mae:.4f}")
    print(f"整体RMSE: {rmse:.4f}")

    # 选取5个代表性OD对进行评估
    sample_indices = np.random.choice(y_true.shape[1], 5, replace=False)
    for idx in sample_indices:
        od_mae = mean_absolute_error(y_true[:, idx], y_pred[:, idx])
        od_rmse = np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx]))
        print(f"OD点{idx} - MAE: {od_mae:.4f}, RMSE: {od_rmse:.4f}")

    return mae, rmse


print("\n模型评估结果:")
mae, rmse = evaluate_predictions(y_test, y_pred)


# 10. 可视化部分结果
def visualize_results(y_true, y_pred, sample_index=0, od_index=0):
    """可视化单个OD对的预测结果"""
    plt.figure(figsize=(15, 6))
    plt.plot(y_true[sample_index, :, od_index], 'b-', label='真实值')
    plt.plot(y_pred[sample_index, :, od_index], 'r--', label='预测值')
    plt.title(f"时间点 {sample_index} - OD点 {od_index} 的预测对比")
    plt.xlabel("时间步")
    plt.ylabel("出租车需求量")
    plt.legend()
    plt.show()


# 恢复空间结构
y_test_spatial = y_test.reshape(y_test.shape[0], 75, 75, -1)
y_pred_spatial = y_pred.reshape(y_pred.shape[0], 75, 75, -1)

print("\n生成可视化...")
visualize_results(y_test_spatial, y_pred_spatial, sample_index=100, od_index=150)

# 11. 保存模型预测结果
# if not DEMO_MODE:
#     print("保存预测结果...")
#     np.save('svr_od_predictions.npy', {
#         'true': y_test_spatial,
#         'pred': y_pred_spatial,
#         'time_features': X_test[:, -4:],
#         'pca': pca,
#     })