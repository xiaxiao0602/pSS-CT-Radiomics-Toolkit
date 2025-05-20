import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import xgboost as xgb
from hyperopt import tpe, hp, fmin, Trials, STATUS_OK
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime
import joblib
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import json

# 定义数据与结果文件路径
file_path_1 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_1.csv'
file_path_2 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_2.csv'
file_path_3 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_3.csv'
file_path_4 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_4.csv'


def preprocess_data(X, y):
    # 初始化分层五折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    # 初始化标准化器
    scaler = MinMaxScaler()

    # 遍历每一折
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        # 根据索引划分训练集和测试集
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 仅对第一折进行处理后返回
        if fold == 0:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # 返回第一折的数据和标准化器后，立即退出函数
            return X_train, X_test, y_train, y_test, scaler


def get_model_and_params(model_name):
    """返回指定模型及其参数空间"""
    if model_name == 'XGBoost':
        param_space = {
            'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
            'eta': hp.loguniform('eta', np.log(0.0001), np.log(0.5)),
            'gamma': hp.uniform('gamma', 0, 10),
            'lambda': hp.uniform('lambda', 1, 100),
            'max_depth': hp.quniform('max_depth', 3, 25, 1),
            'min_child_weight': hp.uniform('min_child_weight', 1, 100),
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 10),
            'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 10)
        }
        model = xgb.XGBClassifier
        
    elif model_name == 'LightGBM':
        param_space = {
            'num_leaves': hp.quniform('num_leaves', 8, 128, 1),
            'max_depth': hp.quniform('max_depth', 3, 8, 1),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-3), np.log(10.0)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-3), np.log(10.0)),
            'min_child_samples': hp.quniform('min_child_samples', 5, 50, 1),
            'min_gain_to_split': hp.loguniform('min_gain_to_split', np.log(1e-5), np.log(1e-1)),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 0.9),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 0.9),
            'bagging_freq': hp.quniform('bagging_freq', 1, 7, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
            'n_estimators': hp.quniform('n_estimators', 100, 500, 10),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 50, 1),
            'max_bin': hp.quniform('max_bin', 200, 300, 10)
        }
        model = lgb.LGBMClassifier
        
    elif model_name == 'LogisticRegression':
        param_space = {
            'C': hp.loguniform('C', np.log(1e-3), np.log(1e3)),
            'max_iter': hp.quniform('max_iter', 5000, 15000, 100),
            'solver': hp.choice('solver', ['lbfgs']),
            'tol': hp.loguniform('tol', np.log(1e-5), np.log(1e-2)),
            'penalty': hp.choice('penalty', ['l2']),
            'class_weight': hp.choice('class_weight', ['balanced', None]),
            'warm_start': hp.choice('warm_start', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }
        model = LogisticRegression
        
    elif model_name == 'AdaBoost':
        param_space = {
            'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.05), np.log(0.5)),
            'algorithm': 'SAMME.R',
            'random_state': 42
        }
        model = AdaBoostClassifier
        
    elif model_name == 'GBDT':
        param_space = {
            'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
            'max_depth': hp.quniform('max_depth', 3, 6, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
            'subsample': hp.uniform('subsample', 0.7, 1.0)
        }
        model = GradientBoostingClassifier
        
    elif model_name == 'GNB':
        param_space = {
            'priors': [0.5, 0.5],
            'var_smoothing': hp.loguniform('var_smoothing', np.log(1e-8), np.log(1e-6))
        }
        model = GaussianNB
        
    elif model_name == 'MLP':
        param_space = {
            # 网络结构
            'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [
                (50,), (100,), (150,), (200,), (300,),  # 单层
                (50, 25), (100, 50), (150, 75), (200, 100), (300, 150),  # 两层
                (100, 50, 25), (200, 100, 50), (300, 150, 75), (400, 200, 100),  # 三层
                (200, 150, 100, 50), (300, 200, 100, 50)  # 四层
            ]),
            
            # 激活函数
            'activation': hp.choice('activation', ['relu', 'tanh', 'logistic']),
            
            # 优化器选择
            'solver': hp.choice('solver', ['adam', 'sgd']), 
            
            # adam优化器参数
            'learning_rate_init': hp.loguniform('learning_rate_init', np.log(0.00001), np.log(0.1)),
            'beta_1': hp.uniform('beta_1', 0.8, 0.999),
            'beta_2': hp.uniform('beta_2', 0.9, 0.9999),
            
            # 学习率调度
            'learning_rate': hp.choice('learning_rate', ['constant', 'adaptive', 'invscaling']),  
            'power_t': hp.uniform('power_t', 0.1, 0.9),  
            
            # 正则化参数
            'alpha': hp.loguniform('alpha', np.log(1e-7), np.log(1e-1)),
            
            # 动量参数（仅在solver='sgd'时有效）
            'momentum': hp.uniform('momentum', 0.1, 0.9),
            'nesterovs_momentum': hp.choice('nesterovs_momentum', [True, False]),
            
            # 训练参数
            'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
            'max_iter': hp.choice('max_iter', [200, 400, 600, 800, 1000, 1500, 2000]),
            'shuffle': True,  # 设为固定值
            
            # 早停机制
            'early_stopping': True,
            'validation_fraction': hp.uniform('validation_fraction', 0.1, 0.3),
            'n_iter_no_change': hp.choice('n_iter_no_change', [10, 20, 30, 40, 50]),
            'tol': hp.loguniform('tol', np.log(1e-6), np.log(1e-2)),
            
            # 固定参数
            'random_state': 42,
            'verbose': False
        }
        model = MLPClassifier
        
    elif model_name == 'SVM':
        param_space = {
            'kernel': 'rbf',
            
            # 正则化参数
            'C': hp.loguniform('C', np.log(0.1), np.log(10.0)),
            
            # rbf核的参数
            'gamma': hp.loguniform('gamma', np.log(0.001), np.log(0.1)),
            
            # 必须设置为True以获取概率预测
            'probability': True,
            
            # 其他固定参数
            'random_state': 42,
            'cache_size': 1000
        }
        model = SVC
        
    elif model_name == 'RF':
        param_space = {
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 10),
            'max_depth': hp.quniform('max_depth', 3, 25, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 4, 1),
            'max_features': hp.choice('max_features', ['sqrt', 'log2']),
            'bootstrap': hp.choice('bootstrap', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestClassifier

    return model, param_space


def train_and_evaluate_model(params, model_class, X, y, cv=5, scaler=None):
    """训练评估函数"""
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=123)
    aucs = []
    
    # 如果是LightGBM，处理参数类型
    if isinstance(model_class(), lgb.LGBMClassifier):
        # 添加一些基础参数
        base_params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbose': -1,  # 减少输出
            'force_col_wise': True,  # 强制按列训练
            'deterministic': True,  # 确保结果可重现
            'boost_from_average': True
        }
        params.update(base_params)
        
        # 需要转换为整数的参数
        int_params = ['num_leaves', 'max_depth', 'min_child_samples', 
                     'bagging_freq', 'n_estimators', 'min_data_in_leaf', 'max_bin']
        for param in int_params:
            if param in params:
                params[param] = int(round(params[param]))

    # 如果是RandomForest，处理参数类型
    if isinstance(model_class(), RandomForestClassifier):
        # 需要转换为整数的参数
        int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
        for param in int_params:
            if param in params:
                params[param] = int(round(params[param]))

    
    # 读取增强数据
    aug_data_folder = 'data/VAE_aug_2'
    
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # 读取对应折的增强数据
        aug_file = os.path.join(aug_data_folder, f'fold_{fold}_train_aug.csv')
        if os.path.exists(aug_file):
            # 读取增强数据
            aug_data = pd.read_csv(aug_file)
            X_aug = aug_data.iloc[:, 1:-1].values  # 跳过Name列和label列
            y_aug = aug_data['label'].values

            # 对增强数据进行标准化
            X_aug = scaler.transform(X_aug)  # 使用与训练数据相同的scaler进行标准化
            
            # 合并原始训练数据和增强数据
            X_train = np.vstack((X_train, X_aug))
            y_train = np.concatenate((y_train, y_aug))
        
        try:
            model = model_class(**params)
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, probabilities)
            aucs.append(auc_score)
            
        except Exception as e:
            print(f"训练出错: {e}")
            return {'loss': np.inf, 'status': STATUS_OK}
    
    avg_auc = np.mean(aucs)
    return {'loss': -avg_auc, 'status': STATUS_OK}


def hyperparameter_optimization(model_name, X, y, max_evals=30, scaler=None):
    """超参数优化函数"""
    model_class, param_space = get_model_and_params(model_name)
    
    def objective(params):
        # 在传递给模型之前转换整数参数
        params_int = params.copy()
        int_params = ['n_estimators', 'max_depth', 'num_leaves', 'max_iter', 
                     'min_samples_split', 'min_child_samples', 'batch_size',
                     'n_iter_no_change']
        
        for param in int_params:
            if param in params:
                params_int[param] = int(params[param])
        
        # 对于MLP的hidden_layer_sizes，确保所有值都是整数
        if 'hidden_layer_sizes' in params:
            if isinstance(params['hidden_layer_sizes'], tuple):
                params_int['hidden_layer_sizes'] = tuple(int(x) for x in params['hidden_layer_sizes'])
        
        return train_and_evaluate_model(params_int, model_class, X, y, scaler=scaler)
    
    trials = Trials()
    best = fmin(objective, param_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    
    # 处理最终的最佳参数
    best_int = best.copy()
    int_params = ['n_estimators', 'max_depth', 'num_leaves', 'max_iter', 
                  'min_samples_split', 'min_child_samples', 'batch_size',
                  'n_iter_no_change']
    
    for param in int_params:
        if param in best:
            best_int[param] = int(best[param])
    
    print(f"Current best parameters for {model_name}: {best_int}")
    print(f"Current best loss: {-trials.best_trial['result']['loss']}")
    
    return best_int, trials


if __name__ == '__main__':
    data_1 = pd.read_csv(file_path_1, index_col=0)
    data_2 = pd.read_csv(file_path_2, index_col=0)
    data_3 = pd.read_csv(file_path_3, index_col=0)
    data_4 = pd.read_csv(file_path_4, index_col=0)
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
    # 将y转换为0和1
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    # 标准化
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    RESULT_PATH = r'param_other_models_aug'
    os.makedirs(RESULT_PATH, exist_ok=True)
    # 定义要训练的模型列表
    models = ['SVM', 'LightGBM', 'LogisticRegression', 'RF', 'XGBoost',
              'AdaBoost', 'GBDT',  'MLP']
    model_epoches = {
        'SVM': 200,
        'LightGBM': 200,
        'LogisticRegression': 200,
        'RF': 100,
        'XGBoost': 200,
        'AdaBoost': 50,
        'GBDT': 5,
        'MLP': 50
    }

    best_params_all = {}
    for model_name in models:
        try:
            print(f"\nTraining {model_name}...")
            # 为每个模型创建单独的结果目录
            model_result_path = os.path.join(RESULT_PATH, model_name)
            os.makedirs(model_result_path, exist_ok=True)
            
            # 训练和优化模型
            epoches = model_epoches[model_name]
            best_params, trials = hyperparameter_optimization(model_name, X, y, max_evals=epoches, scaler=scaler)
            best_params_all[model_name] = best_params
            
            # 记录调参结果
            results = []
            for trial in trials.trials:
                results.append({
                    'params': trial['misc']['vals'],
                    'loss': trial['result']['loss'],
                    'time': datetime.now()
                })
            
            # 保存参数结果
            params_file = os.path.join(model_result_path, "params_results.csv")
            pd.DataFrame(results).to_csv(params_file)
        
        except Exception as e:
            print(f"训练{model_name}出错: {e}")
            continue
    
    # 保存所有模型的最佳参数
    best_params_file = os.path.join(RESULT_PATH, "best_params_all.json")
    with open(best_params_file, 'w') as f:
        json.dump(best_params_all, f)
    
    # # 训练最终模型
    # model_class, _ = get_model_and_params(model_name)
    
    # # 创建最终模型
    # if model_name == 'LogisticRegression':
    #     # 转换solver参数
    #     solvers = ['lbfgs', 'newton-cg', 'sag', 'saga']
    #     final_params = {
    #         'C': float(best_params['C']),
    #         'max_iter': int(best_params['max_iter']),
    #         'solver': solvers[int(best_params['solver'])],
    #         'tol': float(best_params['tol']),
    #         'penalty': best_params['penalty'],
    #         'random_state': 42,
    #         'class_weight': best_params['class_weight'],
    #         'n_jobs': -1
    #     }
    #     final_model = LogisticRegression(**final_params)
    # else:
    #     final_model = model_class(**best_params)
    
    # if isinstance(final_model, SVC):
    #     final_model.probability = True
    
    # # 训练最终模型
    # final_model.fit(X_train, y_train)
    
    # # 保存模型
    # model_file = os.path.join(model_result_path, "final_model.pkl")
    # joblib.dump(final_model, model_file)
    # print(f"{model_name} model saved to: {model_file}")
    
    # # 评估模型
    # y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    # print(f"{model_name} Test AUC: {roc_auc_score(y_test, y_pred_proba)}")


    
