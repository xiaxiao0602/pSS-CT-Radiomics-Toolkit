import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

# 导入各种模型
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class MultiModelEvaluator:
    def __init__(self, model_name, model_params=None):
        # 初始化单个模型
        self.models = {
            'XGBoost': xgb.XGBClassifier,
            'LightGBM': LGBMClassifier,
            'LogisticRegression': LogisticRegression,
            'AdaBoost': AdaBoostClassifier,
            'GBDT': GradientBoostingClassifier,
            'GNB': GaussianNB,
            'MLP': MLPClassifier,
            'SVM': SVC,
            'RF': RandomForestClassifier
        }
        
        if model_name not in self.models:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.model_name = model_name
        self.model_class = self.models[model_name]
        
        # 设置默认参数
        self.default_params = {
            'XGBoost': {'random_state': 42},
            'LightGBM': {'random_state': 42},
            'LogisticRegression': {'random_state': 42},
            'AdaBoost': {'random_state': 42},
            'GBDT': {'random_state': 42},
            'GNB': {},
            'MLP': {'random_state': 42},
            'SVM': {'random_state': 42, 'probability': True},
            'RF': {'random_state': 42}
        }
        
        # 更新模型参数
        self.model_params = self.default_params[model_name].copy()
        if model_params:
            self.model_params.update(model_params)

    def load_and_preprocess_data(self, file_paths):
        """按照 models_param_unilateral.py 的方式加载数据"""
        # 读取数据
        data_list = [pd.read_csv(path, index_col=0) for path in file_paths]
        
        # 提取特征
        X_1 = data_list[0].iloc[:, 22:-1]
        X_2 = data_list[1].iloc[:, 22:-1]
        X_3 = data_list[2].iloc[:, 22:-1]
        X_4 = data_list[3].iloc[:, 22:-1]

        # 添加前缀
        X_1.columns = ['parotid_' + col for col in X_1.columns]
        X_2.columns = ['parotid_' + col for col in X_2.columns]
        X_3.columns = ['submandibular_' + col for col in X_3.columns]
        X_4.columns = ['submandibular_' + col for col in X_4.columns]

        # 合并数据
        X_group_1 = pd.concat([X_1, X_3], axis=1)
        X_group_2 = pd.concat([X_2, X_4], axis=1)
        X = pd.concat([X_group_1, X_group_2], axis=0)
        
        # 处理标签
        y_group_1 = data_list[0].iloc[:, -1]
        y_group_2 = data_list[1].iloc[:, -1]
        y = pd.concat([y_group_1, y_group_2], axis=0)
        
        # 标签编码
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        
        return X, y

    def evaluate_model(self, X, y, n_splits=5, save_path=None):
        """评估单个模型"""
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
        model_metrics = {
            'auc': [], 'accuracy': [], 
            'precision': [], 'recall': [], 'f1': []
        }
        
        print(f"\n评估模型: {self.model_name}")
        
        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            # 数据分割
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # 标准化
            scaler = MinMaxScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

            # 读取对应折的增强数据
            aug_data_folder = 'data/VAE_aug'
            aug_file = os.path.join(aug_data_folder, f'fold_{fold}_train_aug.csv')
            if os.path.exists(aug_file):
                # 读取增强数据
                aug_data = pd.read_csv(aug_file)
                X_aug = aug_data.iloc[:, 1:-1]  # 跳过Name列和label列
                y_aug = aug_data['label'].values
                
                # 确保增强数据的列名与原始数据相同
                X_aug.columns = X_train.columns
                
                # 对增强数据进行标准化，保持DataFrame格式
                X_aug = pd.DataFrame(scaler.transform(X_aug), columns=X_train.columns)
                
                # 合并原始训练数据和增强数据
                X_train = pd.concat([X_train, X_aug], axis=0)
                y_train = np.concatenate((y_train, y_aug))
            
            # 模型训练
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train)
            
            # 预测和评估
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_val)[:, 1]
                model_metrics['auc'].append(roc_auc_score(y_val, y_prob))
            else:
                y_prob = model.predict(X_val)
            
            y_pred = model.predict(X_val)
            model_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            model_metrics['precision'].append(precision_score(y_val, y_pred))
            model_metrics['recall'].append(recall_score(y_val, y_pred))
            model_metrics['f1'].append(f1_score(y_val, y_pred))
            
            # 保存模型
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                model_dict = {"model": model, "scaler": scaler}
                joblib.dump(model_dict, 
                          os.path.join(save_path, f'{self.model_name}_fold_{fold}.pkl'))
                
                print("y_val:", y_val)
                # 打印y_val中0和1的数量
                print("y_val中0和1的数量:", np.bincount(y_val))
            
            print(f"Fold {fold} - AUC: {model_metrics['auc'][-1]:.4f}")
        
        # 计算平均指标
        results = {
            metric: np.mean(scores) 
            for metric, scores in model_metrics.items()
        }
        
        print(f"\n{self.model_name} 平均性能指标:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        
        return results

# 使用示例
if __name__ == "__main__":
    # 设置路径
    data_paths = [
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_1.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_2.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_3.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_4.csv'
    ]
    
    # 为不同模型定义最优参数
    model_params_dict = {
        "SVM": {
            "C": 8.796520258275661,
            "gamma": 0.006008238211202232
        },
        "LightGBM": {
            "subsample": 0.8842754450529282,
            "subsample_freq": 2,
            "colsample_bytree": 0.6072170520179903,
            "learning_rate": 0.04782124162107611,
            "max_bin": 240,
            "max_depth": 6,
            "min_child_samples": 42,
            "min_split_gain": 0.0005498471000824319,
            "n_estimators": 440,
            "num_leaves": 84,
            "reg_alpha": 0.006820919649535423,
            "reg_lambda": 0.0163227779583561
        },
        "LogisticRegression": {
            'C': 1.1525058495205858, 
            'class_weight': 'balanced', 
            'max_iter': 14900, 
            'penalty': 'l2', 
            'solver': 'lbfgs', 
            'tol': 0.0017270487847566415, 
            'warm_start': True
        },
        "RF": {
            "bootstrap": False,
            "max_depth": 10,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "n_estimators": 220
        },
        "XGBoost": {
            "colsample_bynode": 0.8109449415597878,
            "colsample_bytree": 0.7210548910183924,
            "eta": 0.015340705079299456,
            "gamma": 0.7517069313247466,
            "lambda": 3.9897649324862883,
            "max_depth": 17,
            "min_child_weight": 4.471767688987242,
            "n_estimators": 710,
            "scale_pos_weight": 6.101198213014049
        },
        "AdaBoost": {
            "learning_rate": 0.10399618120167037,
            "n_estimators": 280
        },
        "GBDT": {
            "learning_rate": 0.11017688496715419,
            "max_depth": 6,
            "min_samples_split": 3,
            "n_estimators": 180,
            "subsample": 0.9417077620959825
        },
        "MLP": {
            "activation": "tanh",
            "alpha": 0.00013591023764574433,
            "batch_size": "auto",
            "beta_1": 0.951750839895007,
            "beta_2": 0.9642320938862191,
            "hidden_layer_sizes": [9],
            "learning_rate": "invscaling",
            "learning_rate_init": 0.0018244202246032477,
            "max_iter": 4,
            "momentum": 0.8059542107999648,
            "n_iter_no_change": 3,
            "nesterovs_momentum": False,
            "power_t": 0.5102307386807746,
            "solver": "lbfgs",
            "tol": 1.0059949411184408e-06,
            "validation_fraction": 0.245761614149819
        }
    }
    
    # 定义要评估的模型列表
    # models = ['XGBoost', 'LightGBM', 'LogisticRegression', 'AdaBoost', 
    #           'GBDT', 'MLP', 'SVM', 'RF']
    models = ['LightGBM']

    # 选择要评估的模型
    for model_name in models:
        # 获取对应模型的参数
        model_params = model_params_dict[model_name]
        
        # 初始化评估器
        evaluator = MultiModelEvaluator(model_name, model_params)
        
        # 加载和预处理数据
        X, y = evaluator.load_and_preprocess_data(data_paths)
        
        save_folder = r'other_KFold_model_aug'
        save_path = os.path.join(save_folder, f'{model_name}')
        os.makedirs(save_path, exist_ok=True)
        # 评估模型
        results = evaluator.evaluate_model(X, y, save_path=save_path)
