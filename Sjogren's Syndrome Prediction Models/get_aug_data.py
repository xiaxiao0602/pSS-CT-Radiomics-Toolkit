import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import argparse
from VAE import main as vae_main
from tensorboard import program

def load_and_preprocess_data():
    # 定义数据文件路径
    file_path_1 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_1.csv'
    file_path_2 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_2.csv'
    file_path_3 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_3.csv'
    file_path_4 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_4.csv'
    
    # 读取数据
    data_1 = pd.read_csv(file_path_1, index_col=0)
    data_2 = pd.read_csv(file_path_2, index_col=0)
    data_3 = pd.read_csv(file_path_3, index_col=0)
    data_4 = pd.read_csv(file_path_4, index_col=0)
    
    # 提取特征和标签
    X_1 = data_1.iloc[:, 22:-1]
    X_2 = data_2.iloc[:, 22:-1]
    X_3 = data_3.iloc[:, 22:-1]
    X_4 = data_4.iloc[:, 22:-1]

    X_group_1 = pd.concat([X_1, X_3], axis=1)
    X_group_2 = pd.concat([X_2, X_4], axis=1)
    X = pd.concat([X_group_1, X_group_2], axis=0)
    
    y_group_1 = data_1.iloc[:, -1]
    y_group_2 = data_2.iloc[:, -1]
    y = pd.concat([y_group_1, y_group_2], axis=0)
    
    # 标签编码
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    # # 特征标准化
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)
    
    return X, y

def generate_fold_data(X, y, fold_idx, train_index, val_index):
    """为指定折生成训练数据文件"""
    # 创建临时CSV文件
    temp_dir = 'data/temp'
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, f'fold_{fold_idx}_train.csv')
    
    # 获取训练集数据
    X_train = X.iloc[train_index]    # 修改：使用 iloc 进行索引
    y_train = y[train_index]
    
    # 获取验证集数据
    X_val = X.iloc[val_index]        # 修改：使用 iloc 进行索引
    y_val = y[val_index]

    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    
    return X_train, X_val, y_train, y_val

def main():
    # 添加TensorBoard启动代码
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'runs'])
    url = tb.launch()
    print(f"TensorBoard is on: {url}")
    
    # 创建输出目录
    # VAE_aug_2: correct reconstruction_loss
    # VAE_aug: wrong    
    output_dir = os.path.join('data', 'VAE_aug_2')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    X, y = load_and_preprocess_data()
    
    # 初始化StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    
    # 对每一折进行处理
    for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"\n处理第 {fold_idx} 折...")
        
        # 生成当前折的训练数据文件
        train_features, val_features, train_labels, val_labels = generate_fold_data(X, y, fold_idx, train_index, val_index)
        
        # 运行VAE进行数据增强
        vae_main(
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            config_path='config.yaml',
            save_folder=output_dir
        )
        
        # 将生成的两个label的文件合并，并重命名为fold_{}_train_aug.csv
        generated_file_1 = os.path.join(output_dir, 'generated_features_0.csv')
        generated_file_2 = os.path.join(output_dir, 'generated_features_1.csv')
        generated_file_path = os.path.join(output_dir, f'fold_{fold_idx}_train_aug.csv')
        # 检查文件是否存在
        if os.path.exists(generated_file_1) and os.path.exists(generated_file_2):
            # 读取CSV文件
            df1 = pd.read_csv(generated_file_1)
            df2 = pd.read_csv(generated_file_2)
            # 合并数据框
            generated_file = pd.concat([df1, df2], axis=0)
            # 保存合并后的文件
            generated_file.to_csv(generated_file_path, index=False)
            # 安全删除原文件
            try: 
                os.remove(generated_file_1)
                os.remove(generated_file_2)
            except OSError as e:
                print(f"删除文件时出错: {e}")
        else:
            print(f"警告：未找到生成的特征文件")

        print(f"第 {fold_idx} 折数据增强完成")


if __name__ == '__main__':
    main()
