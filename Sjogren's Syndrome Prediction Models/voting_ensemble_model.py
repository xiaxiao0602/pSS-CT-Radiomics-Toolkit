import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier
import joblib
from itertools import product
import json
from datetime import datetime
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


class VotingEnsembleModel:
    def __init__(self, voting='soft', weights=None, random_state=123):
        self.voting = voting
        self.weights = weights
        self.random_state = random_state
        
    def load_and_preprocess_data(self, file_paths):
        """加载和预处理数据"""
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

    def load_pretrained_models(self, base_model_path, fold):
        """加载预训练模型"""
        model_paths = {
            'XGBoost': os.path.join(base_model_path, 'XGBoost', f'XGBoost_fold_{fold}.pkl'),
            'LightGBM': os.path.join(base_model_path, 'LightGBM', f'LightGBM_fold_{fold}.pkl'),
            'LogisticRegression': os.path.join(base_model_path, 'LogisticRegression', f'LogisticRegression_fold_{fold}.pkl'),
            'AdaBoost': os.path.join(base_model_path, 'AdaBoost', f'AdaBoost_fold_{fold}.pkl'),
            'GBDT': os.path.join(base_model_path, 'GBDT', f'GBDT_fold_{fold}.pkl'),
            # 'GNB': os.path.join(base_model_path, 'GNB', f'GNB_fold_{fold}.pkl'),
            'MLP': os.path.join(base_model_path, 'MLP', f'MLP_fold_{fold}.pkl'),
            'SVM': os.path.join(base_model_path, 'SVM', f'SVM_fold_{fold}.pkl'),
            'RF': os.path.join(base_model_path, 'RF', f'RF_fold_{fold}.pkl')
        }
        
        estimators = []
        scalers = {}
        
        for name, path in model_paths.items():
            if os.path.exists(path):
                model_dict = joblib.load(path)
                estimators.append((name, model_dict['model']))
                scalers[name] = model_dict['scaler']
            else:
                print(f"警告: 找不到模型 {path}")
                
        return estimators, scalers

    def create_voting_model(self, estimators):
        """创建Voting集成模型"""
        if isinstance(self.weights, dict):
            weights = [self.weights.get(name, 1.0) for name, _ in estimators]
        else:
            weights = self.weights

        return VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            weights=weights,
            n_jobs=-1
        )
    
    def evaluate_model(self, X, y, pretrained_models_path, save_path=None, n_splits=5):
        """评估Voting集成模型"""
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # 存储模型的结果
        model_metrics = {
            'auc': [], 'accuracy': [], 
            'precision': [], 'recall': [], 'f1': []
        }
        
        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            print(f"\n评估 Fold {fold + 1}/{n_splits}")
            
            # 加载预训练模型和对应的scaler
            estimators, scalers = self.load_pretrained_models(pretrained_models_path, fold)
            
            # 获取验证集
            X_val = X.iloc[val_index]
            y_val = y[val_index]
            
            # 使用对应的scaler转换验证集，并保持特征名称
            first_scaler = list(scalers.values())[0]
            X_val_scaled = pd.DataFrame(
                first_scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )
            
            # 创建和使用Voting模型
            voting_model = self.create_voting_model(estimators)
            
            # 手动设置某些属性
            voting_model.estimators_ = [est for name, est in estimators]
            voting_model.named_estimators_ = {name: est for name, est in estimators}
            voting_model.classes_ = np.array([0, 1])  # 假设是二分类问题
            voting_model.le_ = LabelEncoder().fit(y)  # 手动设置le_
            
            # 预测和评估
            if self.voting == 'soft':
                y_prob = voting_model.predict_proba(X_val_scaled)[:, 1]
            else:  # hard voting
                y_prob = np.mean([model.predict_proba(X_val_scaled)[:, 1] 
                                for name, model in voting_model.named_estimators_.items()
                                if hasattr(model, 'predict_proba')], axis=0)
            
            y_pred = voting_model.predict(X_val_scaled)
            
            # 计算指标
            model_metrics['auc'].append(roc_auc_score(y_val, y_prob))
            model_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            model_metrics['precision'].append(precision_score(y_val, y_pred))
            model_metrics['recall'].append(recall_score(y_val, y_pred))
            model_metrics['f1'].append(f1_score(y_val, y_pred))
            
            # 保存voting模型
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                model_dict = {
                    "model": voting_model,
                    "scaler": first_scaler
                }
                joblib.dump(model_dict, 
                          os.path.join(save_path, f'voting_ensemble_fold_{fold}.pkl'))
            
            print(f"Fold {fold + 1} - AUC: {model_metrics['auc'][-1]:.4f}")
        
        # 计算平均指标
        results = {
            metric: {
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            for metric, scores in model_metrics.items()
        }
        
        print("\nVoting Ensemble 平均性能指标:")
        for metric, stats in results.items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return results
    
    def search_best_weights(self, X, y, pretrained_models_path, save_path, n_splits=5, max_evals=100):
        """使用hyperopt搜索最佳权重组合"""
        # 定义搜索空间
        space = {
            'RF': hp.quniform('RF', 1, 5, 0.5),
            'XGBoost': hp.quniform('XGBoost', 1, 5, 0.5),
            'GBDT': hp.quniform('GBDT', 1, 5, 0.5),
            'MLP': hp.quniform('MLP', 1, 5, 0.5),
            'LightGBM': hp.quniform('LightGBM', 1, 5, 0.5),
            'SVM': hp.quniform('SVM', 1, 5, 0.5),
            'AdaBoost': hp.quniform('AdaBoost', 1, 5, 0.5),
            'LogisticRegression': hp.quniform('LogisticRegression', 1, 5, 0.5)
        }

        trials = Trials()
        best_weights = None
        best_auc = 0
        best_model_results = None
        
        # 创建日志文件
        log_file = os.path.join(save_path, f'weight_search_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

        def objective(weights):
            # 更新模型权重
            self.weights = weights
            
            # 评估当前权重组合
            results = self.evaluate_model(X, y, pretrained_models_path, None, n_splits)
            mean_auc = results['auc']['mean']
            
            # 记录到日志
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"权重组合: {weights}\n")
                f.write(f"平均 AUC: {mean_auc:.4f}\n")
                f.write("-" * 50 + "\n")
            
            # 如果是最佳结果，保存模型
            nonlocal best_auc, best_weights, best_model_results
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_weights = weights.copy()
                best_model_results = results
                self.evaluate_model(X, y, pretrained_models_path, save_path, n_splits)
            
            return {'loss': -mean_auc, 'status': STATUS_OK}

        # 运行优化
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )

        # 保存搜索历史
        search_history = []
        for trial in trials.trials:
            if trial['result']['status'] == STATUS_OK:
                search_history.append({
                    'weights': trial['misc']['vals'],
                    'mean_auc': -trial['result']['loss']
                })

        with open(os.path.join(save_path, 'search_history.json'), 'w', encoding='utf-8') as f:
            json.dump(search_history, f, ensure_ascii=False, indent=4)
        
        # 保存最佳权重
        with open(os.path.join(save_path, 'best_weights.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'best_weights': best_weights,
                'best_auc': best_auc,
                'best_results': best_model_results
            }, f, ensure_ascii=False, indent=4)
        
        return best_weights, best_auc, best_model_results

if __name__ == "__main__":
    # 设置路径
    data_paths = [
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_1.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_2.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_3.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_4.csv'
    ]
    
    # 预训练模型路径
    pretrained_models_path = 'other_KFold_model'
        
    # 创建保存路径
    save_folder = 'other_KFold_model'
    save_path = os.path.join(save_folder, 'voting_ensemble')
    os.makedirs(save_path, exist_ok=True)
    
    voting_ensemble = VotingEnsembleModel(voting='soft', random_state=123)
    X, y = voting_ensemble.load_and_preprocess_data(data_paths)
    
    # 搜索最佳权重
    best_weights, best_auc, best_results = voting_ensemble.search_best_weights(
        X, y,
        pretrained_models_path=pretrained_models_path,
        save_path=save_path,
        n_splits=5
    )
    
    print("\n最佳权重组合:")
    print(best_weights)
    print(f"\n最佳平均 AUC: {best_auc:.4f}")
