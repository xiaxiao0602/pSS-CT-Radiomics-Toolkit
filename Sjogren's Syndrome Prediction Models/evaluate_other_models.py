import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from predict_bagging_ensemble import BaggingEnsemblePredictor
import traceback

# 添加 ModelWrapper 类定义
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

class ModelEvaluator:
    def __init__(self, model_name, data_paths, save_base_path):
        self.model_name = model_name
        self.data_paths = data_paths
        self.save_base_path = os.path.join(save_base_path, model_name)
        os.makedirs(self.save_base_path, exist_ok=True)
        
    def load_data(self):
        """加载数据"""
        data_list = [pd.read_csv(path, index_col=0) for path in self.data_paths]
        
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

    def plot_roc_curve(self, y_test, y_scores, fold, save_path):
        """绘制ROC曲线"""
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(dpi=600)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.title(f'ROC curve of {self.model_name} (Fold {fold})')
        plt.legend(loc="lower right")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return fpr, tpr, roc_auc

    def plot_confusion_matrix(self, y_test, y_pred, fold, save_path):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 7), dpi=600, constrained_layout=True)
        cax = ax.matshow(cm, cmap='Blues')
        fig.colorbar(cax)
        
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()
        
        ax.set_xlabel('Predicted', fontsize=20)
        ax.set_ylabel('Actual', fontsize=20)
        ax.set_xticks(np.arange(2))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(['Class 0', 'Class 1'], fontsize=16)
        ax.set_yticklabels(['Class 0', 'Class 1'], fontsize=16)
        
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f'{val}', ha='center', va='center', color='black', fontsize=24)
            
        plt.title(f'Confusion Matrix - {self.model_name} (Fold {fold})')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, model, X_train, X_test, short_feature_names, fold, save_path):
        """统一使用SHAP值计算特征重要性"""
        try:
            # 生成SHAP值
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 取阳性类别的SHAP值
            
            # 计算SHAP重要性
            importance = pd.DataFrame({
                'feature': short_feature_names,
                'importance': np.abs(shap_values).mean(0)
            }).sort_values('importance', ascending=False)
            
            # 绘制特征重要性图
            plt.figure(figsize=(12, 6), dpi=600)
            plt.bar(importance['feature'][:10], importance['importance'][:10])
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Feature Importance (SHAP) - {self.model_name} (Fold {fold})')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"无法生成特征重要性图: {e}")

    def plot_shap_analysis(self, model, X_train, X_test, short_feature_names, fold, save_path):
        """绘制SHAP摘要图（使用小提琴图样式）"""
        try:
            # 生成SHAP值
            if self.model_name == 'LogisticRegression':
                explainer = shap.LinearExplainer(model, X_train)
            else:
                explainer = shap.TreeExplainer(model)
            
            shap_values = explainer.shap_values(X_test)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # 计算特征重要性
            importance = np.abs(shap_values).mean(0)
            sorted_idx = np.argsort(importance)[::-1][:10]
            
            filtered_shap = shap_values[:, sorted_idx]
            filtered_features = X_test.iloc[:, sorted_idx]
            filtered_names = [short_feature_names[i] for i in sorted_idx]

            # 设置全局字体
            plt.rcParams['font.family'] = 'Times New Roman'
            
            # 绘制摘要图
            plt.figure(figsize=(12, 8), dpi=600)
            shap.summary_plot(
                filtered_shap,
                filtered_features,
                feature_names=filtered_names,
                plot_type="dot",
                # plot_type="violin",  # 改为小提琴图
                show=False,
                max_display=10
            )
            
            # 修改字体设置
            plt.title(f'Top 10 Feature Impacts - {self.model_name} (Fold {fold})', 
                     fontsize=16, fontname='Times New Roman')
            plt.xlabel('SHAP value (impact on prediction)', 
                      fontsize=14, fontname='Times New Roman')
            
            # 修改y轴标签字体
            plt.gca().set_yticklabels(plt.gca().get_yticklabels(), 
                                     fontname='Times New Roman', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.savefig(save_path[:-4] + '.svg', bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"无法生成SHAP摘要图: {e}")
            if 'shap_values' in locals():
                print(f"SHAP值维度: {np.array(shap_values).shape}")
                print(f"特征数量: {len(short_feature_names)}")
            print(f"输入数据维度: {X_test.shape if 'X_test' in locals() else 'N/A'}")

    def plot_feature_groups_importance(self, model, X_train, X_test, fold, save_path):
        """绘制前后两组特征的SHAP值总体重要性比较"""
        try:
            # 生成SHAP值
            if self.model_name == 'LogisticRegression':
                explainer = shap.LinearExplainer(model, X_train)
            else:
                explainer = shap.TreeExplainer(model)
            
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # 计算特征数量的中点
            mid_point = X_test.shape[1] // 2
            # print(mid_point)
            
            # 计算两组特征的平均重要性
            first_half_importance = np.abs(shap_values[:, :mid_point]).mean()
            second_half_importance = np.abs(shap_values[:, mid_point:]).mean()
            
            # 创建横向柱状图
            plt.figure(figsize=(10, 6), dpi=600)
            plt.rcParams['font.family'] = 'Times New Roman'
            
            # 绘制横向柱状图
            groups = ['Parotid', 'Submandibular']
            importances = [first_half_importance, second_half_importance]
            
            bars = plt.barh(groups, importances, color=['#AFC7EB', '#F9BEB9'])
            
            # # 在柱子上添加数值标签
            # for i, v in enumerate(importances):
            #     plt.text(v, i, f'{v:.4f}', 
            #             va='center', fontsize=12, fontname='Times New Roman')
            
            plt.title(f'Average Feature Importance Comparison\n{self.model_name} (Fold {fold})', 
                     fontsize=16, fontname='Times New Roman')
            plt.xlabel('Average feature importance', fontsize=14, fontname='Times New Roman')
            plt.ylabel('Feature Groups', fontsize=14, fontname='Times New Roman')
            
            # 设置y轴标签字体
            plt.gca().set_yticklabels(groups, fontname='Times New Roman', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            return first_half_importance, second_half_importance
            
        except Exception as e:
            print(f"无法生成特征组重要性比较图: {e}")
            return 0.0, 0.0  # 返回默认值而不是None

    def plot_combined_feature_groups_importance(self, all_fold_importances, save_path):
        """绘制所有fold的特征组重要性比较"""
        try:
            plt.figure(figsize=(8, 6), dpi=600)
            plt.rcParams['font.family'] = 'Times New Roman'
            
            # 定义颜色方案 (使用5种不同的蓝色调)
            colors = [
                '#1f77b4',  # 标准蓝
                '#4292c6',  # 稍浅蓝
                '#6baed6',  # 更浅蓝
                '#9ecae1',  # 浅蓝
                '#c6dbef'   # 最浅蓝
            ]
            
            n_folds = len(all_fold_importances)
            x = np.arange(2)  # 两组特征
            width = 0.14  # 柱子的宽度
            
            # 为每个fold绘制一组柱子
            for i, fold_data in enumerate(all_fold_importances):
                offset = width * (i - (n_folds-1)/2)
                plt.bar(x + offset, 
                       [fold_data['first_half'], fold_data['second_half']], 
                       width,
                       color=colors[i % len(colors)],  # 循环使用颜色
                    #    edgecolor='black',  # 添加黑色边框
                    #    linewidth=0.5,
                       label=f'Fold {i}',
                       alpha=0.8)
            
            plt.title(f'Feature Groups Importance Across All Folds ({self.model_name})', 
                     fontsize=16, fontname='Times New Roman')
            plt.xlabel('Feature Groups', fontsize=14, fontname='Times New Roman')
            plt.ylabel('Mean feature importance', fontsize=14, fontname='Times New Roman')
            
            plt.xticks(x, ['Parotid', 'Submandibular'], 
                      fontname='Times New Roman', fontsize=12)
            
            plt.legend(prop={'family': 'Times New Roman', 'size': 12}, 
                      frameon=True, 
                      framealpha=0.8)
            plt.grid(True, alpha=0.3)
            
            # 调整布局
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.savefig(save_path[:-4] + '.svg', bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"无法生成组合特征组重要性比较图: {e}")

    def plot_multi_instance_shap(self, model, X_test, y_test, y_pred, y_scores, fold, save_path):
        """绘制多个实例的组合SHAP force plot"""
        try:
            # 生成SHAP值
            if self.model_name == 'LogisticRegression':
                explainer = shap.LinearExplainer(model, X_test)
            else:
                explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')

            explanation = explainer(X_test)
            shap_values = explanation.values[..., 1]  # 获取正类的SHAP值
            
            # 按预测分数排序
            sort_idx = np.argsort(y_scores)[::-1]  # 降序排序
            sorted_shap_values = shap_values[sort_idx]
            sorted_features = X_test.iloc[sort_idx]
            
            # 生成并保存HTML格式的交互式图
            force_plot = shap.force_plot(
                base_value=explainer.expected_value[1],
                shap_values=sorted_shap_values,
                features=sorted_features,
                feature_names=X_test.columns,
                ordering_keys=True,
                contribution_threshold=0.2,
                show=False,
                text_rotation=45,
                figsize=(20, 3),
                link='identity'
            )

            # 自定义CSS样式，针对具体的SVG元素
            custom_css = """<style>
                body, div, text, span {
                    font-family: "Times New Roman", Times, serif !important;
                }
                .force-bar-axis text,
                .force-bar-labels text,
                .force-bar-value text,
                .feature-name,
                .feature-value,
                .force-bar-label,
                svg text {
                    font-family: "Times New Roman", Times, serif !important;
                    font-size: 18px !important;
                }
            </style>"""

            # 打开文件，写入内容
            with open(os.path.join(save_path, f'multi_instance_force_plot_fold_{fold}.html'), 'w', encoding='utf-8') as f:
                f.write("<html><head><meta http-equiv='content-type' content='text/html'; charset='utf-8'>")
                f.write(custom_css)  # 在head标签内添加自定义样式
                f.write(shap.getjs())  # 添加必要的JavaScript
                f.write("</head><body>\n")
                f.write(force_plot.html())  # 写入force plot的HTML内容
                f.write("</body></html>\n")
            
        except Exception as e:
            print(f"无法生成多实例SHAP force plot: {e}")
            # print(f"expected_value: {explainer.expected_value}")
            # print(f"shap_values shape: {np.array(shap_values).shape}")
            # print(f"X_test shape: {X_test.shape}")


    def evaluate(self, models_path):
        """评估模型"""
        X, y = self.load_data()
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
        
        fprs, tprs, aucs = [], [], []
        
        # 创建评估指标列表
        evaluation_metrics = []
        
        # 存储所有fold的特征组重要性数据
        all_fold_importances = []
        
        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            print(f"评估 {self.model_name} - Fold {fold}")
            
            # 准备数据
            X_train, X_test = X.iloc[train_index], X.iloc[val_index]
            y_train, y_test = y[train_index], y[val_index]
            
            # 加载模型
            model_path = os.path.join(models_path, f'{self.model_name}_fold_{fold}.pkl')
            model_dict = joblib.load(model_path)
            model = model_dict["model"]
            scaler = model_dict["scaler"]
            
            # 数据标准化
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
            # 转换为DataFrame并保留特征名
            feature_names = X.columns.tolist()
            X_train = pd.DataFrame(X_train, columns=feature_names)
            X_test = pd.DataFrame(X_test, columns=feature_names)
            
            # 创建保存文件夹
            fold_save_path = os.path.join(self.save_base_path, f'fold_{fold}')
            os.makedirs(fold_save_path, exist_ok=True)
            
            # 预测
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                y_scores = model.predict(X_test)
            
            # 找到最佳阈值
            best_threshold = 0
            best_f1 = 0
            for threshold in np.arange(0.01, 1, 0.01):
                y_pred = (y_scores > threshold).astype(int)
                current_f1 = f1_score(y_test, y_pred)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_threshold = threshold
            print(f"最佳阈值: {best_threshold:.2f}, F1分数: {best_f1:.4f}")
            
            # 使用最佳阈值进行预测
            y_pred = (y_scores > best_threshold).astype(int)
            
            # 生成简短特征名
            short_feature_names = [f'F{i}' for i in range(len(feature_names))]
            
            # 保存特征映射
            with open(os.path.join(fold_save_path, 'feature_mapping.txt'), 'w') as f:
                for i, name in enumerate(feature_names):
                    f.write(f"F{i}: {name}\n")
            
            # 绘制并保存评估图
            fpr, tpr, roc_auc = self.plot_roc_curve(
                y_test, y_scores, fold, 
                os.path.join(fold_save_path, 'roc_curve.png')
            )
            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(roc_auc)
            
            self.plot_confusion_matrix(
                y_test, y_pred, fold,
                os.path.join(fold_save_path, 'confusion_matrix.png')
            )
            
            self.plot_feature_importance(
                model, X_train, X_test, short_feature_names, fold,
                os.path.join(fold_save_path, 'feature_importance.png')
            )
            
            self.plot_shap_analysis(
                model, X_train, X_test, short_feature_names, fold,
                os.path.join(fold_save_path, 'shap_analysis.png')
            )
            
            # 修改plot_feature_groups_importance函数以返回重要性值
            first_half_importance, second_half_importance = self.plot_feature_groups_importance(
                model, X_train, X_test, fold,
                os.path.join(fold_save_path, 'feature_groups_importance.png')
            )
            
            # 收集每个fold的数据
            all_fold_importances.append({
                'first_half': first_half_importance,
                'second_half': second_half_importance
            })
            
            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # 计算各项指标
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # 保存该折的评估指标
            fold_metrics = {
                'fold': fold,
                'auc': roc_auc,
                'best_threshold': best_threshold,
                'f1_score': best_f1,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'accuracy': accuracy,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            }
            evaluation_metrics.append(fold_metrics)
            
            # 添加多实例SHAP分析
            self.plot_multi_instance_shap(
                model, X_test, y_test, y_pred, y_scores, fold,
                fold_save_path
            )
        
        # 绘制组合图
        self.plot_combined_feature_groups_importance(
            all_fold_importances,
            os.path.join(self.save_base_path, 'combined_feature_groups_importance.png')
        )
        
        # 计算平均值和标准差
        metrics_df = pd.DataFrame(evaluation_metrics)
        mean_metrics = metrics_df.mean(numeric_only=True)
        std_metrics = metrics_df.std(numeric_only=True)
        # 添加总体统计行
        summary_metrics = pd.concat([
            metrics_df,
            pd.DataFrame({
                'fold': ['Mean'],
                'auc': [mean_metrics['auc']],
                'best_threshold': [mean_metrics['best_threshold']],
                'f1_score': [mean_metrics['f1_score']],
                'sensitivity': [mean_metrics['sensitivity']],
                'specificity': [mean_metrics['specificity']],
                'ppv': [mean_metrics['ppv']],
                'npv': [mean_metrics['npv']],
                'accuracy': [mean_metrics['accuracy']],
                'tp': [mean_metrics['tp']],
                'tn': [mean_metrics['tn']],
                'fp': [mean_metrics['fp']],
                'fn': [mean_metrics['fn']]
            }),
            pd.DataFrame({
                'fold': ['Std'],
                'auc': [std_metrics['auc']],
                'best_threshold': [std_metrics['best_threshold']],
                'f1_score': [std_metrics['f1_score']],
                'sensitivity': [std_metrics['sensitivity']],
                'specificity': [std_metrics['specificity']],
                'ppv': [std_metrics['ppv']],
                'npv': [std_metrics['npv']],
                'accuracy': [std_metrics['accuracy']],
                'tp': [std_metrics['tp']],
                'tn': [std_metrics['tn']],
                'fp': [std_metrics['fp']],
                'fn': [std_metrics['fn']]
            })
        ])
        
        # 保存评估指标到CSV文件
        metrics_save_path = os.path.join(self.save_base_path, 'evaluation_metrics.csv')
        summary_metrics.to_csv(metrics_save_path, index=False)
        
        # 绘制组合ROC曲线
        plt.figure(figsize=(8, 6))
        for i in range(n_splits):
            plt.plot(fprs[i], tprs[i], 
                    label=f'ROC fold {i} (AUC = {aucs[i]:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', 
                label='Chance', alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Combined ROC Curves - {self.model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.save_base_path, 'combined_roc_curve.png'))
        plt.close()

    def evaluate_bagging_ensemble(self, selected_models=None):
        """评估bagging集成模型"""
        print("\n评估 Bagging 集成模型")
        X, y = self.load_data()
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
        
        # 初始化评估指标列表
        evaluation_metrics = []
        fprs, tprs, aucs = [], [], []
        
        # 初始化bagging预测器
        predictor = BaggingEnsemblePredictor(model_path='bagging_models')
        
        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            print(f"评估 Bagging Ensemble - Fold {fold}")
            
            # 准备数据
            X_test = X.iloc[val_index]
            y_test = y[val_index]
            
            # 使用bagging集成进行预测
            y_scores, y_pred = predictor.ensemble_predict(
                data=X_test,
                selected_models=selected_models,
                n_folds=5  # 使用所有fold的模型
            )
            
            # 找到最佳阈值
            best_threshold = 0
            best_f1 = 0
            for threshold in np.arange(0.01, 1, 0.01):
                y_pred_temp = (y_scores > threshold).astype(int)
                current_f1 = f1_score(y_test, y_pred_temp)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_threshold = threshold
            
            # 使用最佳阈值进行预测
            y_pred = (y_scores > best_threshold).astype(int)
            
            # 创建保存文件夹
            fold_save_path = os.path.join(self.save_base_path, f'fold_{fold}')
            os.makedirs(fold_save_path, exist_ok=True)
            
            # 绘制ROC曲线
            fpr, tpr, roc_auc = self.plot_roc_curve(
                y_test, y_scores, fold,
                os.path.join(fold_save_path, 'roc_curve.png')
            )
            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(roc_auc)
            
            # 绘制混淆矩阵
            self.plot_confusion_matrix(
                y_test, y_pred, fold,
                os.path.join(fold_save_path, 'confusion_matrix.png')
            )
            
            # 计算混淆矩阵指标
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # 计算各项指标
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # 保存该折的评估指标
            fold_metrics = {
                'fold': fold,
                'auc': roc_auc,
                'best_threshold': best_threshold,
                'f1_score': best_f1,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'accuracy': accuracy,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            }
            evaluation_metrics.append(fold_metrics)
            
        # 计算平均值和标准差
        metrics_df = pd.DataFrame(evaluation_metrics)
        mean_metrics = metrics_df.mean(numeric_only=True)
        std_metrics = metrics_df.std(numeric_only=True)
        
        # 添加总体统计行
        summary_metrics = pd.concat([
            metrics_df,
            pd.DataFrame({'fold': ['Mean'], **{k: [v] for k, v in mean_metrics.items()}}),
            pd.DataFrame({'fold': ['Std'], **{k: [v] for k, v in std_metrics.items()}})
        ])
        
        # 保存评估指标
        metrics_save_path = os.path.join(self.save_base_path, 'bagging_ensemble', 'evaluation_metrics.csv')
        summary_metrics.to_csv(metrics_save_path, index=False)
        
        # 绘制组合ROC曲线
        plt.figure(figsize=(8, 6))
        for i in range(n_splits):
            plt.plot(fprs[i], tprs[i], 
                    label=f'ROC fold {i} (AUC = {aucs[i]:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', 
                label='Chance', alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Combined ROC Curves - Bagging Ensemble')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.save_base_path, 'bagging_ensemble', 'combined_roc_curve.png'))
        plt.close()
        
        return summary_metrics

if __name__ == "__main__":
    # 设置路径
    data_paths = [
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_1.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_2.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_3.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_4.csv'
    ]
    
    # save_base_path = r'D:\Desktop\parotid_XGBoost\other_model_evaluations'
    # os.makedirs(save_base_path, exist_ok=True)
    # models_base_path = r'D:\Desktop\parotid_XGBoost\other_KFold_model'
    
    save_base_path = r'D:\Desktop\parotid_XGBoost\other_model_evaluations_aug'
    os.makedirs(save_base_path, exist_ok=True)
    models_base_path = r'D:\Desktop\parotid_XGBoost\other_KFold_model_aug'
    

    # 定义要评估的模型列表
    # models = ['XGBoost', 'LightGBM', 'LogisticRegression', 'AdaBoost', 
    #           'GBDT', 'MLP', 'SVM', 'RF']
    models = ['RF']

    # save_base_path = r'D:\Desktop\parotid_XGBoost\other_model_evaluations_aug'
    # os.makedirs(save_base_path, exist_ok=True)
    # models_base_path = r'stacking_models_aug\20250220'
    # models = ['stacking_model'] 
    
    # 选择要评估的模型
    
    # model_name = 'voting_ensemble'  # 可以改为其他模型名称
    for model_name in models:
        # 创建评估器并运行评估
        evaluator = ModelEvaluator(model_name, data_paths, save_base_path)
        evaluator.evaluate(os.path.join(models_base_path, model_name)) 
        # evaluator.evaluate(models_base_path) 
    
    # # 添加bagging集成模型的评估
    # selected_models = ['XGBoost', 'LightGBM', 'LogisticRegression', 'AdaBoost', 
    #                    'GBDT', 'GNB', 'MLP', 'SVM', 'RF']  # 选择要集成的模型
    # evaluator = ModelEvaluator('bagging_ensemble', data_paths, save_base_path)
    # results = evaluator.evaluate_bagging_ensemble(selected_models)
    
    # print("\nBagging集成模型评估结果:")
    # print(results) 