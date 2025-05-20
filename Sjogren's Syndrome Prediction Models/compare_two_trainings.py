import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelPerformanceComparator:
    def __init__(self, base_path1, base_path2, save_path):
        """
        初始化比较器
        base_path1: 第一次训练的模型评估结果根目录
        base_path2: 第二次训练的模型评估结果根目录
        save_path: 保存结果的路径
        """
        self.base_path1 = base_path1
        self.base_path2 = base_path2
        self.save_path = save_path
        self.models = ['voting_ensemble', 'RF', 'LightGBM', 'XGBoost', 'GBDT', 'LogisticRegression', 
                       'stacking_model', 'SVM', 'AdaBoost', 'MLP']
        os.makedirs(save_path, exist_ok=True)

    def read_model_results(self, base_path):
        """
        读取所有模型的评估结果
        返回格式: {model_name: [auc_values]}
        """
        results = {}
        for model in self.models:
            metrics_file = os.path.join(base_path, model, 'evaluation_metrics.csv')
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                if 'auc' in df.columns:  # 确保AUC列存在
                    # 只读取每个fold的AUC值，不包括Mean和Std行
                    aucs = df[df['fold'].apply(lambda x: str(x).isdigit())]['auc'].values
                    results[model] = aucs
        return results

    def plot_comparison(self):
        """绘制对比箱线图"""
        # 读取两次训练的结果
        results1 = self.read_model_results(self.base_path1)
        results2 = self.read_model_results(self.base_path2)

        # 准备数据框
        data = []
        for model in results1.keys():
            if model in results2:
                # 第一次训练的数据
                for auc in results1[model]:
                    data.append({
                        'Model': 'Voting' if 'voting' in model.lower() else ('Stacking' if 'stacking' in model.lower() else model),
                        'AUC': auc,
                        'Training': 'With CVAE data'
                    })
                # 第二次训练的数据
                for auc in results2[model]:
                    data.append({
                        'Model': 'Voting' if 'voting' in model.lower() else ('Stacking' if 'stacking' in model.lower() else model),
                        'AUC': auc,
                        'Training': 'Original data'
                    })

        df = pd.DataFrame(data)

        # 创建箱线图
        plt.figure(figsize=(12, 6), dpi=300)
        plt.rcParams['font.family'] = 'Times New Roman'
        
        # 使用seaborn绘制箱线图，明确指定图例标签
        ax = sns.boxplot(x='Model', y='AUC', hue='Training', data=df,
                        palette=['#AFC7EB', '#F9BEB9'],
                        width=0.6)

        plt.xticks(rotation=45, fontsize=12)
        plt.xlabel('Model Name', fontsize=14)
        plt.ylabel('AUC', fontsize=14)
        plt.title('Performance Comparison of Machine Learning Models', fontsize=16)
        
        # 修改图例
        handles = ax.legend_.legend_handles
        labels = ['With CVAE data', 'Original data']
        plt.legend(handles, labels, prop={'family': 'Times New Roman', 'size': 12})
        
        plt.grid(True, alpha=0.3)
        
        # 自动调整y轴范围，让差异更明显
        auc_min = df['AUC'].min()
        auc_max = df['AUC'].max()
        plt.ylim(max(0.8, auc_min - 0.02), min(1.0, auc_max + 0.02))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'training_comparison_boxplot.png'))
        plt.savefig(os.path.join(self.save_path, 'training_comparison_boxplot.svg'))
        plt.close()

        # 保存统计信息
        self.save_statistical_analysis(df)

    def save_statistical_analysis(self, df):
        """保存统计分析结果"""
        with open(os.path.join(self.save_path, 'statistical_analysis.txt'), 'w', encoding='utf-8') as f:
            f.write("模型性能统计分析\n")
            f.write("-" * 80 + "\n\n")
            
            for model in df['Model'].unique():
                f.write(f"模型: {model}\n")
                f.write("-" * 40 + "\n")
                
                # 第一次训练的统计
                train1_stats = df[(df['Model'] == model) & 
                                (df['Training'] == 'First Training')]['AUC'].describe()
                f.write("第一次训练:\n")
                f.write(f"平均 AUC: {train1_stats['mean']:.4f}\n")
                f.write(f"标准差: {train1_stats['std']:.4f}\n")
                f.write(f"最小值: {train1_stats['min']:.4f}\n")
                f.write(f"最大值: {train1_stats['max']:.4f}\n\n")
                
                # 第二次训练的统计
                train2_stats = df[(df['Model'] == model) & 
                                (df['Training'] == 'Second Training')]['AUC'].describe()
                f.write("第二次训练:\n")
                f.write(f"平均 AUC: {train2_stats['mean']:.4f}\n")
                f.write(f"标准差: {train2_stats['std']:.4f}\n")
                f.write(f"最小值: {train2_stats['min']:.4f}\n")
                f.write(f"最大值: {train2_stats['max']:.4f}\n\n")
                
                # 计算改进
                improvement = train2_stats['mean'] - train1_stats['mean']
                f.write(f"性能提升: {improvement:+.4f}\n")
                f.write("\n" + "=" * 80 + "\n\n")

if __name__ == "__main__":
    # 设置路径
    base_path1 = "other_model_evaluations_aug"    # 第一次训练的结果目录
    base_path2 = "other_model_evaluations"   # 第二次训练的结果目录
    save_path = "comparison_CVAE_results"                 # 保存结果的文件夹
    
    # 创建比较器并生成对比图
    comparator = ModelPerformanceComparator(base_path1, base_path2, save_path)
    comparator.plot_comparison() 