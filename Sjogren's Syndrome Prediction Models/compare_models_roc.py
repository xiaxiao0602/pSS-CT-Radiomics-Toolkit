import os
import joblib
import numpy as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from stacking_model import ModelWrapper
from predict_bagging_ensemble import BaggingEnsemblePredictor

class ModelComparator:
    def __init__(self, data_paths, models_base_path, save_path):
        self.data_paths = data_paths
        self.models_base_path = models_base_path
        self.save_path = save_path
        self.models = ['XGBoost', 'LightGBM', 'LogisticRegression', 
                      'AdaBoost', 'GBDT', 'MLP', 'SVM', 'RF'
                      , 'voting_ensemble'
                      ]
        # 'stacking_model', 'voting_ensemble'
        # 为每个模型定义不同的颜色
        self.colors = {
            'XGBoost': '#FF4500',      # 橙红色
            'LightGBM': '#32CD32',     # 绿色
            'LogisticRegression': '#4169E1',  # 蓝色
            'AdaBoost': '#FFD700',     # 金色
            'GBDT': '#8B008B',         # 紫色
            'GNB': '#FF69B4',          # 粉色
            'MLP': '#20B2AA',          # 青色
            'SVM': '#FF8C00',          # 深橙色
            'RF': '#800000',           # 褐红色
            'stacking_model': '#483D8B', # 暗蓝色
            'bagging_ensemble': '#006400',  # 深绿色
            'voting_ensemble': '#8B4513'  # 马鞍棕色
        }
        os.makedirs(save_path, exist_ok=True)
        self.model_aucs = {model: [] for model in self.models}

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

    def plot_fold_comparison(self, fold, X, y, train_index, val_index):
        """为特定折绘制所有模型的ROC曲线"""
        plt.figure(figsize=(10, 8), dpi=300)
        
        # 准备数据
        X_train, X_test = X.iloc[train_index], X.iloc[val_index]
        y_train, y_test = y[train_index], y[val_index]
        
        # 初始化bagging预测器
        bagging_predictor = BaggingEnsemblePredictor(model_path='bagging_models')
        selected_models = ['XGBoost', 'LightGBM', 'RF']  # 选择要集成的模型
        
        for model_name in self.models:
            try:
                if model_name == 'bagging_ensemble':
                    # 使用bagging集成模型进行预测
                    y_scores, _ = bagging_predictor.ensemble_predict(
                        data=X_test,
                        selected_models=selected_models,
                        n_folds=5
                    )
                else:
                    # 加载其他模型
                    model_path = os.path.join(self.models_base_path, model_name, 
                                            f'{model_name}_fold_{fold}.pkl')
                    model_dict = joblib.load(model_path)
                    model = model_dict["model"]
                    scaler = model_dict["scaler"]
                    
                    # 数据标准化
                    X_test_scaled = scaler.transform(X_test)
                    
                    # 获取预测概率
                    if hasattr(model, "predict_proba"):
                        y_scores = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        y_scores = model.predict(X_test_scaled)
                
                # 计算ROC曲线
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                
                # 存储AUC值
                self.model_aucs[model_name].append(roc_auc)
                
                # 绘制ROC曲线
                plt.plot(fpr, tpr, color=self.colors[model_name], lw=2,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
                
            except Exception as e:
                print(f"处理模型 {model_name} 时出错: {e}")
        
        # 绘制随机猜测的基准线
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
        
        # 设置图形属性
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC Curves Comparison - Fold {fold}')
        plt.legend(loc="lower right", fontsize='small')
        plt.grid(True, alpha=0.3)
        
        # 保存图形
        plt.savefig(os.path.join(self.save_path, f'roc_comparison_fold_{fold}.png'),
                   bbox_inches='tight')
        plt.close()

    def save_average_aucs(self):
        """计算并保存平均AUC值"""
        # 计算每个模型的平均AUC
        average_aucs = {}
        for model_name, aucs in self.model_aucs.items():
            if aucs:  # 确保有AUC值
                avg_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                average_aucs[model_name] = (avg_auc, std_auc)

        # 按平均AUC值降序排序
        sorted_models = sorted(average_aucs.items(), 
                             key=lambda x: x[1][0], 
                             reverse=True)

        # 保存到文件
        with open(os.path.join(self.save_path, 'average_aucs.txt'), 'w') as f:
            f.write("Model Average AUC Values (Sorted by Performance):\n")
            f.write("-" * 50 + "\n")
            f.write("Model Name".ljust(20) + "Average AUC".ljust(15) + "Standard Deviation\n")
            f.write("-" * 50 + "\n")
            
            for model_name, (avg_auc, std_auc) in sorted_models:
                f.write(f"{model_name.ljust(20)}{f'{avg_auc:.4f}'.ljust(15)}{std_auc:.4f}\n")

    def compare_models(self):
        """比较所有模型在所有折上的表现"""
        X, y = self.load_data()
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        
        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            print(f"Processing Fold {fold}...")
            self.plot_fold_comparison(fold, X, y, train_index, val_index)
        
        # 保存平均AUC值
        self.save_average_aucs()
        self.plot_model_distributions()
        print("All fold comparisons and average AUC values have been generated!")

    def plot_model_distributions(self):
        """Plot sorted box plot and violin plot to show AUC distributions of all models"""
        # 准备数据
        data = []
        labels = []
        for model_name, aucs in self.model_aucs.items():
            if aucs:
                data.append(aucs)
                labels.append(model_name)
        
        # 计算每个模型的平均AUC用于排序
        mean_aucs = [np.mean(aucs) for aucs in data]
        # 获取排序索引
        sort_idx = np.argsort(mean_aucs)[::-1]  # 降序排列
        
        # 对数据进行排序
        data = [data[i] for i in sort_idx]
        labels = [labels[i] for i in sort_idx]
        sorted_colors = [self.colors[label] for label in labels]
        
        # 绘制箱线图
        plt.figure(figsize=(15, 8), dpi=300)
        bp = plt.boxplot(data, patch_artist=True)
        for patch, color in zip(bp['boxes'], sorted_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
        plt.ylabel('AUC Score')
        plt.title('Model Performance Comparison (Box Plot) - Sorted by Mean AUC')
        plt.grid(True, alpha=0.3)
        plt.ylim(0.8, 1.0)  # 设置Y轴范围为0.8-1
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'model_performance_boxplot_sorted.png'))
        plt.close()
        
        # 绘制小提琴图
        import seaborn as sns
        plt.figure(figsize=(15, 8), dpi=300)
        violin_data = []
        for model_name, aucs in zip(labels, data):
            violin_data.extend([(model_name, auc) for auc in aucs])
        violin_df = pd.DataFrame(violin_data, columns=['Model', 'AUC'])
        
        # 设置模型顺序
        model_order = labels
        sns.violinplot(data=violin_df, x='Model', y='AUC', 
                      order=model_order,  # 指定顺序
                      palette={model: self.colors[model] for model in labels})
        
        plt.xticks(rotation=45)
        plt.ylabel('AUC Score')
        plt.title('Model Performance Comparison (Violin Plot) - Sorted by Mean AUC')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'model_performance_violin_sorted.png'))
        plt.close()

if __name__ == "__main__":
    # 设置路径
    data_paths = [
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_1.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_2.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_3.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_4.csv'
    ]
    
    models_base_path = r'D:\Desktop\parotid_XGBoost\other_KFold_model_aug'
    save_path = r'other_model_evaluations_aug\model_comparisons'
    os.makedirs(save_path, exist_ok=True)
    
    # 创建比较器并运行比较
    comparator = ModelComparator(data_paths, models_base_path, save_path)
    comparator.compare_models() 