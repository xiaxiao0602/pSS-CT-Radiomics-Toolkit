import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import json
import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 加载已保存的模型
def load_models(model_paths, fold):
    models = {}
    for name, path_template in model_paths.items():
        path = path_template.format(fold=fold)
        if os.path.exists(path):
            try:
                model_dict = joblib.load(path)
                models[name] = {
                    'model': model_dict['model'],
                    'scaler': model_dict['scaler']
                }
            except Exception as e:
                print(f"Error loading model {name} from {path}: {e}")
        else:
            print(f"Model file not found: {path}")
    return models

# 2. 创建一个包装器类来处理缩放和预测
class ModelWrapper:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        
    def fit(self, X, y):
        # 先对数据进行拟合和变换
        X_scaled = self.scaler.fit_transform(X)
        # 训练模型
        self.model.fit(X_scaled, y)
        # 从基础模型获取类别信息
        self.classes_ = self.model.classes_
        return self
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_params(self, deep=True):
        return {
            "model": self.model,
            "scaler": self.scaler
        }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# 3. 数据加载和预处理
def load_and_preprocess_data(file_paths):
    data_list = [pd.read_csv(path, index_col=0) for path in file_paths]
    
    X_1 = data_list[0].iloc[:, 22:-1]
    X_2 = data_list[1].iloc[:, 22:-1]
    X_3 = data_list[2].iloc[:, 22:-1]
    X_4 = data_list[3].iloc[:, 22:-1]

    X_1.columns = ['parotid_' + col for col in X_1.columns]
    X_2.columns = ['parotid_' + col for col in X_2.columns]
    X_3.columns = ['submandibular_' + col for col in X_3.columns]
    X_4.columns = ['submandibular_' + col for col in X_4.columns]

    X_group_1 = pd.concat([X_1, X_3], axis=1)
    X_group_2 = pd.concat([X_2, X_4], axis=1)
    X = pd.concat([X_group_1, X_group_2], axis=0)
    
    y_group_1 = data_list[0].iloc[:, -1]
    y_group_2 = data_list[1].iloc[:, -1]
    y = pd.concat([y_group_1, y_group_2], axis=0)
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    return X, y

# 4. 主函数
def run_stacking(data_paths, model_paths, save_dir, aug_data_folder, model_config):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载和预处理数据
    X, y = load_and_preprocess_data(data_paths)
    
    # 初始化分层五折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    
    # 存储所有折的指标
    all_metrics = {
        'accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1': []
    }
    
    # 收集所有折的训练信息
    all_folds_info = {
        'model_config': model_config,
        'folds_info': [],
        'average_metrics': {},
        'training_timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 评估每一折
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"\nProcessing fold {fold}")
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 标准化
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        # 加载当前折的模型
        loaded_models = load_models(model_paths, fold)
        
        # 根据配置选择基础模型
        base_models = []
        for model_name in model_config['base_models']:
            if model_name in loaded_models:
                base_models.append(
                    (model_name, ModelWrapper(loaded_models[model_name]['model'], 
                                            loaded_models[model_name]['scaler']))
                )
            else:
                print(f"警告: 配置中指定的模型 {model_name} 未在model_paths中找到")
        
        # 读取增强数据
        aug_file = os.path.join(aug_data_folder, f'fold_{fold}_train_aug.csv')
        if os.path.exists(aug_file):
            # 读取增强数据
            aug_data = pd.read_csv(aug_file)
            X_aug = aug_data.iloc[:, 1:-1]  # 跳过Name列和label列
            y_aug = aug_data['label'].values
            
            # 确保增强数据的列名与原始数据相同
            X_aug.columns = X_train.columns
            
            # 对增强数据进行标准化
            X_aug = pd.DataFrame(scaler.transform(X_aug), columns=X_train.columns)
            
            # 合并原始训练数据和增强数据
            X_train = pd.concat([X_train, X_aug], axis=0)
            y_train = np.concatenate((y_train, y_aug))
        
        # 根据配置创建元模型
        meta_model = model_config['meta_model']['model'](**model_config['meta_model']['params'])
        
        # 创建堆叠模型
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            **model_config['stacking_params']
        )
        
        # 训练堆叠模型
        stacking_model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = stacking_model.predict(X_test)
        y_pred_proba = stacking_model.predict_proba(X_test)[:, 1]
        
        # 计算评估指标
        fold_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        # 存储指标
        for metric, value in fold_metrics.items():
            all_metrics[metric].append(value)
            
        # 打印当前折的评估指标
        print(f"\nFold {fold} Metrics:")
        for metric, value in fold_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # 收集当前折的训练信息
        fold_info = {
            'fold': fold,
            'data_shape': {
                'train': X_train.shape,
                'test': X_test.shape
            },
            'augmented_data_used': os.path.exists(aug_file),
            'metrics': fold_metrics,
            'feature_names': list(X_train.columns)
        }
        all_folds_info['folds_info'].append(fold_info)
        
        # 保存模型（只保存模型和scaler）
        model_save_path = os.path.join(save_dir, f'stacking_model_fold_{fold}.pkl')
        save_dict = {
            "model": stacking_model,
            "scaler": scaler
        }
        joblib.dump(save_dict, model_save_path)
        print(f"Stacking model for fold {fold} saved to: {model_save_path}")
    
    # 计算并保存平均指标
    all_folds_info['average_metrics'] = {
        metric: {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        for metric, values in all_metrics.items()
    }
    
    # 保存配置和训练信息到单个JSON文件
    info_save_path = os.path.join(save_dir, 'stacking_model_info.json')
    # 确保所有数据都是JSON可序列化的
    json_safe_info = {
        'model_config': model_config,
        'folds_info': [
            {k: str(v) if not isinstance(v, (dict, list, int, float, str, bool)) else v
             for k, v in fold_info.items()}
            for fold_info in all_folds_info['folds_info']
        ],
        'average_metrics': all_folds_info['average_metrics'],
        'training_timestamp': all_folds_info['training_timestamp']
    }
    
    with open(info_save_path, 'w') as f:
        json.dump(json_safe_info, f, indent=4)
    print(f"\nModel configuration and training info saved to: {info_save_path}")

# 5. 调用主函数
if __name__ == "__main__":
    # 定义数据路径
    data_paths = [
        r'data\new_data\label_1.csv',
        r'data\new_data\label_2.csv',
        r'data\new_data\label_3.csv',
        r'data\new_data\label_4.csv'
    ]
    
    aug_data_folder = 'data/VAE_aug_2'
    
    # 定义模型路径模板
    model_folder = 'other_KFold_model_aug'
    model_paths = {
        'XGBoost': os.path.join(model_folder, 'XGBoost/XGBoost_fold_{fold}.pkl'),
        'LightGBM': os.path.join(model_folder, 'LightGBM/LightGBM_fold_{fold}.pkl'),
        'LogisticRegression': os.path.join(model_folder, 'LogisticRegression/LogisticRegression_fold_{fold}.pkl'),
        'AdaBoost': os.path.join(model_folder, 'AdaBoost/AdaBoost_fold_{fold}.pkl'),
        'GBDT': os.path.join(model_folder, 'GBDT/GBDT_fold_{fold}.pkl'),
        'MLP': os.path.join(model_folder, 'MLP/MLP_fold_{fold}.pkl'),
        'SVM': os.path.join(model_folder, 'SVM/SVM_fold_{fold}.pkl'),
        'RF': os.path.join(model_folder, 'RF/RF_fold_{fold}.pkl')
    }
    
    # 定义模型配置
    model_config = {
        'base_models': ['XGBoost', 'LightGBM', 'GBDT', 'LogisticRegression', 'AdaBoost', 'RF', 'SVM', 'MLP'],
        'meta_model': {
            'model': RandomForestClassifier,  
            'params': {
                'n_estimators': 500,
                'max_depth': 20,
                'random_state': 42
            }
        },
        'stacking_params': {
            'cv': 5,
            'stack_method': 'predict_proba',
            'passthrough': False
        }
    }
    
    # 定义保存路径
    save_dir = r'stacking_models_aug/20250220'
    os.makedirs(save_dir, exist_ok=True)
    
    # 运行堆叠模型
    run_stacking(data_paths, model_paths, save_dir, aug_data_folder, model_config)
