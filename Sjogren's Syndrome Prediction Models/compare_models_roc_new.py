import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

class ModelROCComparator:
    def __init__(self, base_path, save_path):
        """
        初始化比较器
        base_path: 模型评估结果根目录
        save_path: 保存结果的路径
        """
        self.base_path = base_path
        self.save_path = save_path
        self.models = ['voting_ensemble', 'RF', 'LightGBM', 'XGBoost', 'GBDT', 'LogisticRegression', 
                      'SVM', 'AdaBoost', 'MLP']
        self.colors = {
            'voting_ensemble': '#8B4513',  # 马鞍棕色
            'XGBoost': '#FF4500',      # 橙红色
            'LightGBM': '#32CD32',     # 绿色
            'LogisticRegression': '#4169E1',  # 蓝色
            'AdaBoost': '#FFD700',     # 金色
            'GBDT': '#8B008B',         # 紫色
            'MLP': '#20B2AA',          # 青色
            'SVM': '#FF8C00',          # 深橙色
            'RF': '#800000',           # 褐红色
        }
        os.makedirs(save_path, exist_ok=True)
        self.model_aucs = {model: [] for model in self.models}

    def read_model_results(self):
        """
        读取所有模型的评估结果
        返回格式: {model_name: {'y_true': [...], 'y_pred': [...], 'auc': [...]}}
        """
        results = {}
        for model in self.models:
            metrics_file = os.path.join(self.base_path, model, 'evaluation_metrics.csv')
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                if 'auc' in df.columns:
                    # 只读取每个fold的AUC值，不包括Mean和Std行
                    aucs = df[df['fold'].apply(lambda x: str(x).isdigit())]['auc'].values
                    results[model] = {
                        'auc': aucs,
                        'mean_auc': np.mean(aucs),
                        'std_auc': np.std(aucs)
                    }
        return results

    def plot_roc_curves(self):
        """绘制所有模型的ROC曲线"""
        results = self.read_model_results()
        
        plt.figure(figsize=(10, 8), dpi=300)
        
        # 存储每个模型的平均AUC用于排序
        model_mean_aucs = {}
        
        for model_name, result in results.items():
            mean_auc = result['mean_auc']
            std_auc = result['std_auc']
            model_mean_aucs[model_name] = mean_auc
            
            # 绘制ROC曲线
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
            plt.plot([0, 0, 1], [0, 1, 1], color=self.colors[model_name], lw=2,
                    label=f'{model_name} (AUC = {mean_auc:.3f}±{std_auc:.3f})')
        
        # 设置图形属性
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curves Comparison')
        
        # 根据AUC值排序图例
        handles, labels = plt.gca().get_legend_handles_labels()
        labels_dict = dict(zip(labels, handles))
        
        # 移除Random标签用于排序
        if 'Random' in labels_dict:
            random_handle = labels_dict.pop('Random')
        
        # 根据AUC值排序
        sorted_labels = sorted(labels_dict.keys(), 
                             key=lambda x: model_mean_aucs.get(x.split()[0], 0), 
                             reverse=True)
        
        # 重新添加Random标签到末尾
        if 'Random' in labels:
            sorted_labels.append('Random')
            labels_dict['Random'] = random_handle
        
        # 设置排序后的图例
        plt.legend([labels_dict[label] for label in sorted_labels], 
                  sorted_labels, 
                  loc="lower right", 
                  fontsize='small')
        
        plt.grid(True, alpha=0.3)
        
        # 保存图形
        plt.savefig(os.path.join(self.save_path, 'roc_curves_comparison.png'),
                   bbox_inches='tight')
        plt.close()

        # 保存平均AUC值
        self.save_average_aucs(model_mean_aucs)

    def save_average_aucs(self, model_mean_aucs):
        """保存平均AUC值到文件"""
        # 按AUC值排序
        sorted_models = sorted(model_mean_aucs.items(), key=lambda x: x[1], reverse=True)
        
        with open(os.path.join(self.save_path, 'average_aucs.txt'), 'w') as f:
            f.write("Model Average AUC Values (Sorted by Performance):\n")
            f.write("-" * 50 + "\n")
            f.write("Model Name".ljust(20) + "Average AUC\n")
            f.write("-" * 50 + "\n")
            
            for model_name, avg_auc in sorted_models:
                f.write(f"{model_name.ljust(20)}{avg_auc:.4f}\n")

    def plot_model_distributions(self):
        """绘制模型性能分布的箱线图和小提琴图"""
        results = self.read_model_results()
        
        # 准备数据框
        data = []
        for model_name, result in results.items():
            for auc in result['auc']:
                data.append({
                    'Model': model_name,
                    'AUC': auc
                })
        
        df = pd.DataFrame(data)
        
        # 计算每个模型的平均AUC用于排序
        model_means = df.groupby('Model')['AUC'].mean().sort_values(ascending=False)
        sorted_models = model_means.index.tolist()
        
        # 创建箱线图
        plt.figure(figsize=(12, 6), dpi=300)
        
        # 使用seaborn绘制箱线图
        sns.boxplot(x='Model', y='AUC', data=df, 
                   order=sorted_models,  # 按平均AUC排序
                   palette=self.colors)

        plt.xticks(rotation=45)
        plt.title('Model Performance Comparisons')
        plt.xlabel('Model Name')
        plt.ylabel('AUC')
        plt.grid(True, alpha=0.3)
        
        # 自动调整y轴范围，让差异更明显
        auc_min = df['AUC'].min()
        auc_max = df['AUC'].max()
        plt.ylim(max(0.8, auc_min - 0.02), min(1.0, auc_max + 0.02))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'model_performance_boxplot.png'))
        plt.close()
        
        # 创建小提琴图
        plt.figure(figsize=(12, 6), dpi=300)
        sns.violinplot(x='Model', y='AUC', data=df,
                      order=sorted_models,  # 按平均AUC排序
                      palette=self.colors)
        
        plt.xticks(rotation=45)
        plt.title('Model Performance Distribution (Violin Plot)')
        plt.xlabel('Model Name')
        plt.ylabel('AUC Score')
        plt.grid(True, alpha=0.3)
        plt.ylim(max(0.8, auc_min - 0.02), min(1.0, auc_max + 0.02))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'model_performance_violin.png'))
        plt.close()

    def save_statistical_analysis(self):
        """保存详细的统计分析结果"""
        results = self.read_model_results()
        
        with open(os.path.join(self.save_path, 'statistical_analysis.txt'), 'w', encoding='utf-8') as f:
            f.write("模型性能统计分析\n")
            f.write("-" * 80 + "\n\n")
            
            # 计算并排序平均AUC
            model_stats = {}
            for model_name, result in results.items():
                if result['auc'] is not None:
                    aucs = result['auc']
                    stats = {
                        'mean': np.mean(aucs),
                        'std': np.std(aucs),
                        'min': np.min(aucs),
                        'max': np.max(aucs),
                        'median': np.median(aucs)
                    }
                    model_stats[model_name] = stats
            
            # 按平均AUC排序
            sorted_models = sorted(model_stats.items(), 
                                 key=lambda x: x[1]['mean'], 
                                 reverse=True)
            
            for model_name, stats in sorted_models:
                f.write(f"模型: {model_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"平均 AUC: {stats['mean']:.4f}\n")
                f.write(f"标准差: {stats['std']:.4f}\n")
                f.write(f"最小值: {stats['min']:.4f}\n")
                f.write(f"最大值: {stats['max']:.4f}\n")
                f.write(f"中位数: {stats['median']:.4f}\n")
                f.write("\n" + "=" * 80 + "\n\n")

if __name__ == "__main__":
    # 设置路径
    base_path = r"other_model_evaluations_aug"    # 模型评估结果目录
    save_path = r"other_model_evaluations_aug\model_comparisons_new"          # 保存结果的文件夹
    
    # 创建比较器并生成ROC曲线对比
    comparator = ModelROCComparator(base_path, save_path)
    comparator.plot_roc_curves()
    comparator.plot_model_distributions()
    comparator.save_statistical_analysis() 