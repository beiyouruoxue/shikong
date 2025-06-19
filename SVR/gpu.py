import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
import gc  # 垃圾回收

# 使用cuML替代scikit-learn的SVR
from cuml.svm import SVR  # GPU加速的支持向量回归
import cupy as cp  # 用于显存管理


# 假设数据加载函数
def load_data():
    """模拟加载纽约出租车OD矩阵数据"""
    # 创建随机数据作为示例（实际使用时替换为真实数据）
    return np.load("nyc_taxi.npy")  # 随机数据用于演示
    # return np.load("nyc_taxi.npy")  # 实际使用时取消注释


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

# 释放不再需要的内存
del data, spatial_data
gc.collect()

# 4. 划分训练测试集 (按时间顺序)
test_size = int(0.2 * X.shape[0])
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

# 5. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)  # GPU需要float32
X_test_scaled = scaler.transform(X_test).astype(np.float32)  # GPU需要float32


# 6. 定义多输出SVR封装器（GPU优化版）
class MultiOutputSVR:
    """多输出SVR封装器，处理多目标预测（GPU优化）"""

    def __init__(self,  ** kwargs):
        self.models = []
        self.params = kwargs
        # 设置GPU显存管理参数
        self.batch_size = kwargs.pop('batch_size', 200)
        self.verbose = kwargs.pop('verbose', True)

    def fit(self, X, y):
        """训练多个SVR模型 (每个OD点一个模型)，分批处理以节省显存"""
        num_targets = y.shape[1]
        print(f"训练 {num_targets} 个SVR模型 (批量大小: {self.batch_size})...")
        self.models = []

        # 分批训练
        for batch_start in range(0, num_targets, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_targets)
            batch_models = []

            if self.verbose:
                print(f"训练模型 {batch_start + 1}-{batch_end}/{num_targets}")

            # 训练当前批次的所有模型
            for i in range(batch_start, batch_end):
                model = SVR(**self.params)
                model.fit(X, y[:, i].astype(np.float32))
                batch_models.append(model)

            self.models.extend(batch_models)

            # 显存清理
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            gc.collect()

            if self.verbose:
                # 监控显存使用情况
                mem_info = cp.cuda.runtime.memGetInfo()
                used_gb = (mem_info[1] - mem_info[0]) / (1024 ** 3)
                total_gb = mem_info[1] / (1024 ** 3)
                print(f"GPU显存使用: {used_gb:.2f}GB/{total_gb:.2f}GB")

        return self

    def predict(self, X):
        """生成预测结果，分批处理以节省显存"""
        num_targets = len(self.models)
        predictions = np.zeros((X.shape[0], num_targets), dtype=np.float32)

        # 分批预测
        for batch_start in range(0, num_targets, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_targets)

            if self.verbose:
                print(f"预测模型 {batch_start + 1}-{batch_end}/{num_targets}")

            # 预测当前批次
            for i, model in enumerate(self.models[batch_start:batch_end]):
                idx = batch_start + i
                predictions[:, idx] = model.predict(X)

            # 显存清理
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            gc.collect()

        return predictions


# 7. 训练模型
print("\nTraining model with GPU acceleration...")
start_time = time.time()

# GPU优化参数
params = {
    'kernel': 'rbf',
    'C': 10,
    'epsilon': 0.1,
    'gamma': 'scale',
    'batch_size': 500,  # 显存允许的情况下增大批次大小
    'cache_size': 10000  # 增大缓存提高性能
}

# DEMO模式设置
DEMO_MODE =True  # 设为False使用完整OD矩阵
if DEMO_MODE:
    print("DEMO模式: 仅预测前50个OD点")
    y_train = y_train[:, :50]
    y_test = y_test[:, :50]

# 初始化并训练模型
multi_svr = MultiOutputSVR(**params)
multi_svr.fit(X_train_scaled, y_train)

print(f"训练完成! 总耗时: {time.time() - start_time:.2f}秒")

# 8. 预测测试集
print("Predicting...")
pred_start = time.time()
y_pred = multi_svr.predict(X_test_scaled)
print(f"预测完成! 耗时: {time.time() - pred_start:.2f}秒")


# 9. 评估模型
def evaluate_predictions(y_true, y_pred):
    """评估预测性能"""
    # 总体指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n模型评估结果:")
    print(f"整体MAE: {mae:.4f}")
    print(f"整体RMSE: {rmse:.4f}")

    # 选取5个代表性OD对进行评估
    sample_indices = np.random.choice(y_true.shape[1], 5, replace=False)
    for i, idx in enumerate(sample_indices):
        od_mae = mean_absolute_error(y_true[:, idx], y_pred[:, idx])
        od_rmse = np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx]))
        print(f"OD点{idx} - MAE: {od_mae:.4f}, RMSE: {od_rmse:.4f}")

    return mae, rmse


mae, rmse = evaluate_predictions(y_test, y_pred)


# 10. 可视化部分结果
def visualize_results(y_true, y_pred, sample_index=0, od_index=0):
    """可视化单个OD对的预测结果"""
    plt.figure(figsize=(15, 6))
    plt.plot(y_true[:, od_index], 'b-', label='真实值')
    plt.plot(y_pred[:, od_index], 'r--', label='预测值')
    plt.title(f"OD点 {od_index} 的预测对比 (时间序列)")
    plt.xlabel("时间步")
    plt.ylabel("出租车需求量")
    plt.legend()
    plt.savefig(f"od_prediction_{od_index}.png", dpi=300)
    plt.show()

    # 热力图可视化
    plt.figure(figsize=(10, 8))
    plt.imshow(y_true[sample_index].reshape(75, 75), cmap='hot', interpolation='nearest')
    plt.title(f"时间点 {sample_index} - 真实OD矩阵")
    plt.colorbar()
    plt.savefig("true_od_matrix.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(y_pred[sample_index].reshape(75, 75), cmap='hot', interpolation='nearest')
    plt.title(f"时间点 {sample_index} - 预测OD矩阵")
    plt.colorbar()
    plt.savefig("pred_od_matrix.png", dpi=300)
    plt.show()


# 恢复空间结构
y_test_spatial = y_test.reshape(y_test.shape[0], 75, 75, -1)
y_pred_spatial = y_pred.reshape(y_pred.shape[0], 75, 75, -1)

print("\n生成可视化...")
visualize_results(y_test, y_pred, sample_index=100, od_index=150)

# 11. 保存模型预测结果
if not DEMO_MODE:
    print("保存预测结果...")
    np.save('svr_od_predictions.npy', {
        'true': y_test_spatial,
        'pred': y_pred_spatial,
        'time_features': X_test[:, -4:],
        'pca': pca,
    })