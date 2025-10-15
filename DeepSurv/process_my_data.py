# 简化的数据预处理脚本
import pandas as pd
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split

# 请在这里修改你的CSV文件路径
CSV_FILE_PATH = ""  # 请替换为你的CSV文件路径
DATASET_NAME = "my_data"  # 你可以修改数据集名称

def process_data():
    # 读取CSV
    print("正在读取CSV文件...")
    df = pd.read_csv(CSV_FILE_PATH)
    
    print(f"数据形状: {df.shape}")
    print("前几行数据:")
    print(df.head())
    
    # 获取数据
    columns = df.columns.tolist()
    
    # 第一列是ID（跳过），第二列是事件指示，第三列是时间，其余是特征
    X = df.iloc[:, 3:].values.astype(np.float32)  # 特征（从第4列开始）
    e = df.iloc[:, 1].values.astype(np.int32)     # 事件指示（第2列）
    t = df.iloc[:, 2].values.astype(np.float32)   # 时间（第3列）
    
    # 时间保留两位小数
    #t = np.round(t).astype(np.int32)
    t = np.round(t, 2).astype(np.float32)
    
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    print(f"事件发生数: {np.sum(e)}")
    print(f"时间范围: {np.min(t)} - {np.max(t)}")
    
    # 处理缺失值
    if np.any(np.isnan(X)):
        print("填充缺失值...")
        X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values
    
    # 划分训练测试集
    X_train, X_test, e_train, e_test, t_train, t_test = train_test_split(
        X, e, t, test_size=0.3, random_state=42, stratify=e
    )
    
    # 创建目录
    os.makedirs(f'data/{DATASET_NAME}', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    
    # 保存HDF5
    h5_path = f'data/{DATASET_NAME}/{DATASET_NAME}_survival_data.h5'
    with h5py.File(h5_path, 'w') as f:
        # 训练集
        train_group = f.create_group('train')
        train_group.create_dataset('x', data=X_train)
        train_group.create_dataset('e', data=e_train)
        train_group.create_dataset('t', data=t_train)
        
        # 测试集
        test_group = f.create_group('test')
        test_group.create_dataset('x', data=X_test)
        test_group.create_dataset('e', data=e_test)
        test_group.create_dataset('t', data=t_test)
    
    # 创建配置文件
    config_content = f"""[train]
h5_file = 'data/{DATASET_NAME}/{DATASET_NAME}_survival_data.h5'
epochs = 500
learning_rate = 1e-3
lr_decay_rate = 1e-4
optimizer = 'Adam'

[network]
drop = 0.3
norm = True
dims = [{X.shape[1]}, 32, 16, 1]
activation = 'ReLU'
l2_reg = 1e-4
"""
    
    config_path = f'configs/{DATASET_NAME}.ini'
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\n✅ 处理完成!")
    print(f"HDF5文件: {h5_path}")
    print(f"配置文件: {config_path}")
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    return DATASET_NAME

if __name__ == '__main__':
    # 在运行前，请先修改文件开头的CSV_FILE_PATH变量
    process_data()
