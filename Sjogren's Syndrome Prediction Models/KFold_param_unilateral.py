import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from hyperopt import tpe, hp, fmin, Trials, STATUS_OK
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime
import joblib
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer

# 定义数据与结果文件路径
file_path_1 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_1.csv'
file_path_2 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_2.csv'
file_path_3 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_3.csv'
file_path_4 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_4.csv'
RESULT_PATH = r'D:\Desktop\parotid_XGBoost\param_XGBT_output\new_data\K_Fold_2_unilateral'
os.makedirs(RESULT_PATH, exist_ok=True)
PARAMS_RESULTS_FILE = os.path.join(RESULT_PATH, "params_results.csv")
FINAL_MODEL_FILE = os.path.join(RESULT_PATH, "final_model.pkl")


def preprocess_data(X, y):
    # 初始化分层五折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    # 初始化标准化器
    scaler = StandardScaler()

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


# 更新train_and_evaluate_model函数，使其支持K-Fold CV
def train_and_evaluate_model(params, X, y, cv=5):
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=123)

    aucs = []  # 存储每次CV的AUC值
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 模型训练与预测
        # model = xgb.XGBClassifier(**params)
        model = xgb.XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', **params)
        model.fit(X_train, y_train)

        # 预测概率，用于计算AUC
        probabilities = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, probabilities)
        aucs.append(auc_score)

    avg_auc = sum(aucs) / len(aucs)

    return {'loss': -avg_auc, 'status': STATUS_OK}


# 超参数调优
def hyperparameter_optimization(X, y, max_evals=20):
    param_space = {
        'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
        'eta': hp.loguniform('eta', np.log(0.0001), np.log(0.5)),
        'gamma': hp.uniform('gamma', 0, 10),
        'lambda': hp.uniform('lambda', 1, 100),
        'max_depth': hp.quniform('max_depth', 3, 25, 1),
        'min_child_weight': hp.uniform('min_child_weight', 1, 100),
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 10),
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 10)
    }   # 大范围贝叶斯优化
    # param_space = {
    #     'colsample_bynode': hp.uniform('colsample_bynode', 0.151, 0.154),
    #     'colsample_bytree': hp.uniform('colsample_bytree', 0.47, 0.48),
    #     'eta': hp.loguniform('eta', np.log(0.08), np.log(0.15)),
    #     'gamma': hp.uniform('gamma', 3.3, 3.5),
    #     'lambda': hp.uniform('lambda', 1.3, 1.6),
    #     'max_depth': hp.quniform('max_depth', 11, 18, 1),
    #     'min_child_weight': hp.uniform('min_child_weight', 5, 6),
    #     'n_estimators': hp.quniform('n_estimators', 255, 265, 5),
    #     'objective': 'binary:logistic',
    #     'booster': 'gbtree',
    #     'scale_pos_weight': hp.uniform('scale_pos_weight', 20, 60)
    # }  # 小范围贝叶斯优化, 用于精细调参
    # Current best parameters: {'colsample_bynode': 0.1753176036905465, 'colsample_bytree': 0.984028697385892,
    # 'eta': 0.019823349831983937, 'gamma': 2.2292177084191764, 'lambda': 82.83887780573717, 'max_depth': 21,
    # 'min_child_weight': 20.202359163155513, 'n_estimators': 220, 'scale_pos_weight': 8.733211350879634}

    def objective(params):
        params['n_estimators'] = np.int64(params['n_estimators'])
        params['max_depth'] = np.int64(params['max_depth'])
        params.update({'seed': 1412, 'verbosity': 0, 'n_jobs': 24})
        return train_and_evaluate_model(params, X, y)

    trials = Trials()
    best = fmin(objective, param_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    # 转换需要为整数的参数
    best_int = best.copy()
    best_int['n_estimators'] = int(best['n_estimators'])
    best_int['max_depth'] = int(best['max_depth'])

    # 输出当前最优参数和损失
    print(f"Current best parameters: {best_int}")
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
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    best_params, trials = hyperparameter_optimization(X, y)

    # 记录调参结果
    results = []
    for trial in trials.trials:
        results.append({
            'params': trial['misc']['vals'],
            'loss': trial['result']['loss'],
            'time': datetime.now()
        })

    os.makedirs(RESULT_PATH, exist_ok=True)
    pd.DataFrame(results).to_csv(PARAMS_RESULTS_FILE)

    # 使用最优参数进行最终模型训练
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # 保存模型
    joblib.dump(final_model, FINAL_MODEL_FILE)
    print("Model saved to:", FINAL_MODEL_FILE)

    # 打印测试集的AUC
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    print(f"Test AUC: {roc_auc_score(y_test, y_pred_proba)}")
