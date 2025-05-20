import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            
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
            
            print(f"Fold {fold + 1} - AUC: {model_metrics['auc'][-1]:.4f}")
        
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
        'XGBoost': {
            'colsample_bynode': 0.6037020032096705, 
            'colsample_bytree': 0.5990027750224443, 
            'eta': 0.01969790621800751, 
            'gamma': 1.835900556618535, 
            'lambda': 5.649015480890687, 
            'max_depth': 13, 
            'min_child_weight': 1.0821510852217795, 
            'n_estimators': 420, 
            'scale_pos_weight': 9.659730431590312
        },
        'LightGBM': {
            'colsample_bytree': 0.909281785658207, 
            'learning_rate': 0.21746079710845007, 
            'max_depth': 5, 
            'min_child_samples': 20, 
            'min_child_weight': 0.009706570006028602, 
            'n_estimators': 190, 
            'num_leaves': 40, 
            'reg_alpha': 0.008516135572846978, 
            'reg_lambda': 2.790436294389509e-05, 
            'subsample': 0.648350428212957,
            'random_state': 42,
            'min_split_gain': 0.0,  # 降低分裂增益的阈值
            'min_data_in_leaf': 10,  # 每个叶子节点的最小数据量
            'verbose': -1,  # 减少警告信息
            'boost_from_average': True,
            'force_col_wise': True
        },
        'LogisticRegression': {
            'C': 0.02904039852736716, 
            'max_iter': 13400, 
            'solver': 'sag', 
            'tol': 5.869578423157753e-05
        },
        'AdaBoost': {
            'learning_rate': 0.1123808863097679, 
            'n_estimators': 460
        },
        'GBDT': {
            'learning_rate': 0.1026506972021802, 
            'max_depth': 4, 
            'min_samples_split': 3, 
            'n_estimators': 100, 
            # 'n_estimators': 130, 
            'subsample': 0.9827872039861218
        },
        'GNB': {
            'var_smoothing': 6.131214014669806e-08
        },
        'MLP': {
            'alpha': 0.003264788523304657, 
            'batch_size': 16, 
            'beta_1': 0.8946224803432463, 
            'beta_2': 0.974740338670141, 
            'hidden_layer_sizes': (200, 100), 
            'learning_rate_init': 0.003185349825911041, 
            'max_iter': 400, 
            'n_iter_no_change': 40, 
            'tol': 0.0002531021003387177, 
            'validation_fraction': 0.17139954189115597
        },
        'SVM': {
            'C': 7.0071188492924135, 
            'gamma': 0.0010419038779767773
        },
        'RF': {
            'bootstrap': False, 
            'max_depth': 24, 
            'max_features': 'log2', 
            'min_samples_leaf': 1, 
            'min_samples_split': 3, 
            'n_estimators': 410
        }
    }
    
    # 定义要评估的模型列表
    # models = ['XGBoost', 'LightGBM', 'LogisticRegression', 'AdaBoost', 
    #           'GBDT', 'GNB', 'MLP', 'SVM', 'RF']

    # 选择要评估的模型
    model_name = 'RF'
    
    # 获取对应模型的参数
    model_params = model_params_dict[model_name]
    
    # 初始化评估器
    evaluator = MultiModelEvaluator(model_name, model_params)
    
    # 加载和预处理数据
    X, y = evaluator.load_and_preprocess_data(data_paths)
    
    save_folder = r'other_KFold_model'
    save_path = os.path.join(save_folder, f'{model_name}')
    os.makedirs(save_path, exist_ok=True)
    # 评估模型
    results = evaluator.evaluate_model(X, y, save_path=save_path)
