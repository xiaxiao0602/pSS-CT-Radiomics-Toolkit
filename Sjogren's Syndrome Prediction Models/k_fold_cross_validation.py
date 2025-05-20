import os.path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# 读取数据
file_path_1 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_1.csv'
file_path_2 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_2.csv'
file_path_3 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_3.csv'
file_path_4 = r'D:\Desktop\parotid_XGBoost\data\new_data\label_4.csv'
model_save_path = r'D:\Desktop\parotid_XGBoost\KFold_model\new_data_2_unilateral'
os.makedirs(model_save_path, exist_ok=True)

data_1 = pd.read_csv(file_path_1, index_col=0)
data_2 = pd.read_csv(file_path_2, index_col=0)
data_3 = pd.read_csv(file_path_3, index_col=0)
data_4 = pd.read_csv(file_path_4, index_col=0)

# X_1 = data_1.iloc[:, 22:-1]
# X_2 = data_2.iloc[:, 22:-1]
# X_3 = data_3.iloc[:, 22:-1]
# X_4 = data_4.iloc[:, 22:-1]
X_1 = data_1.iloc[:, [33, 428, 451, 469, 506, 523, 547, 562, 658, 668]]
X_2 = data_2.iloc[:, [33, 428, 451, 469, 506, 523, 547, 562, 658, 668]]
X_3 = data_3.iloc[:, [33, 428, 451, 469, 506, 523, 547, 562, 658, 668]]
X_4 = data_4.iloc[:, [33, 428, 451, 469, 506, 523, 547, 562, 658, 668]]
# 为特征名添加前缀
# X_1.columns = ['parotid_L_' + col for col in X_1.columns]
# X_2.columns = ['parotid_R_' + col for col in X_2.columns]
# X_3.columns = ['submandibular_L_' + col for col in X_3.columns]
# X_4.columns = ['submandibular_R_' + col for col in X_4.columns]
# 合并特征数据
# X = pd.concat([X_1, X_2, X_3, X_4], axis=1)
# y = data_1.iloc[:, -1]


X_1.columns = ['parotid_' + col for col in X_1.columns]
X_2.columns = ['parotid_' + col for col in X_2.columns]
X_3.columns = ['submandibular_' + col for col in X_3.columns]
X_4.columns = ['submandibular_' + col for col in X_4.columns]
# 单侧数据
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
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# 初始化KFold对象
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
# 存储每一轮的结果
auc_scores = []

# 最优参数设定
# best_params = {'colsample_bynode': 0.1753176036905465,
#                'colsample_bytree': 0.984028697385892,
#                'eta': 0.019823349831983937,
#                'gamma': 2.2292177084191764,
#                'lambda': 82.83887780573717,
#                'max_depth': 21,
#                'min_child_weight': 20.202359163155513,
#                'n_estimators': 220,
#                'scale_pos_weight': 8.733211350879634
#                }
# best_params = {'colsample_bynode': 0.16999819710430936,
#                'colsample_bytree': 0.5163448167952599,
#                'eta': 0.060375017525498655,
#                'gamma': 5.928677842782919,
#                'lambda': 58.29326417066897,
#                'max_depth': 16,
#                'min_child_weight': 21.856978403253965,
#                'n_estimators': 550,
#                'scale_pos_weight': 7.533907350982237
#                }  # less_features
best_params = {'colsample_bynode': 0.5677560522512684,
               'colsample_bytree': 0.9236158580607483,
               'eta': 0.02335888447915659,
               'gamma': 1.701059268027355,
               'lambda': 89.7356745594596,
               'max_depth': 15,
               'min_child_weight': 18.058792188936557,
               'n_estimators': 700,
               'scale_pos_weight': 7.107756774327485
               } # 单侧数据训练
# best_params = {'colsample_bynode': 0.1,
#                'colsample_bytree': 0.1,
#                'eta': 0.02335888447915659,
#                'gamma': 25,
#                'lambda': 200,
#                'max_depth': 5,
#                'min_child_weight': 20,
#                'n_estimators': 700,
#                'scale_pos_weight': 7.107756774327485
#                } # test






for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 模型训练与预测
    model = xgb.XGBClassifier(**best_params)

    model.fit(X_train, y_train)


    # 保存模型
    # joblib.dump(model, os.path.join(model_save_path, f'xgb_model_{fold}.pkl'))
    # 保存模型及其对应的scaler
    model_dict = {"model": model, "scaler": scaler}
    joblib.dump(model_dict, os.path.join(model_save_path, f'xgb_model_with_scaler_{fold}.pkl'))

    # 预测概率，用于计算AUC
    probabilities = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, probabilities)
    auc_scores.append(auc_score)

    # print("y_train:", y_train)
    print("y_val:", y_val)
    # 打印y_val中0和1的数量
    print("y_val中0和1的数量:", np.bincount(y_val))
    print(f"Fold {fold + 1} AUC Score: {auc_score}")

# 打印所有fold的平均AUC得分
print("Average AUC over folds:", sum(auc_scores) / len(auc_scores))
