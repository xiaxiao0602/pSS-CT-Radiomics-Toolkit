import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import shap
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler

# 添加 ModelWrapper 类定义，与evaluate_other_models.py中的定义相同
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

class DecisionCurveAnalysis:
    def __init__(self, model_names, data_paths, save_path):
        """
        初始化决策曲线分析类
        
        参数:
        model_names: 要分析的模型名称列表
        data_paths: 数据文件路径列表
        save_path: 结果保存路径
        """
        self.model_names = model_names
        self.data_paths = data_paths
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
    def load_data(self):
        """加载数据"""
        data_list = [pd.read_csv(path, index_col=0) for path in self.data_paths]
        
        # 保存样本名称 - 修改连接方式
        sample_names_1 = pd.Series(data_list[0].index)
        sample_names_2 = pd.Series(data_list[1].index)
        all_sample_names = pd.concat([sample_names_1, sample_names_2]).values
        
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
        
        return X, y, all_sample_names
    
    def calculate_net_benefit(self, y_true, y_pred_proba, threshold):
        """
        计算给定阈值下的净收益
        
        参数:
        y_true: 真实标签
        y_pred_proba: 预测概率
        threshold: 决策阈值
        
        返回:
        net_benefit: 净收益值
        """
        # 特殊情况处理：阈值为1.0时
        if threshold == 1.0:
            # 在阈值为1时，几乎没有样本会被预测为阳性，因此净收益为0
            return 0
            
        # 根据阈值将概率转换为二元预测
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # 计算真阳性率和假阳性率
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)
        
        # 计算净收益
        if np.sum(y_pred) == 0:  # 避免除以零
            return 0
        else:
            return (tp/n) - (fp/n) * (threshold/(1-threshold))
    
    def calculate_model_net_benefit(self, y_true, y_pred_proba, thresholds):
        """
        计算模型在一系列阈值下的净收益
        
        参数:
        y_true: 真实标签
        y_pred_proba: 预测概率
        thresholds: 阈值列表
        
        返回:
        net_benefits: 净收益列表
        """
        net_benefits = []
        for threshold in thresholds:
            net_benefit = self.calculate_net_benefit(y_true, y_pred_proba, threshold)
            net_benefits.append(net_benefit)
        return net_benefits
    
    def calculate_all_positive_net_benefit(self, y_true, thresholds):
        """
        计算"全部治疗"策略的净收益
        
        参数:
        y_true: 真实标签
        thresholds: 阈值列表
        
        返回:
        net_benefits: 净收益列表
        """
        net_benefits = []
        prevalence = np.mean(y_true)  # 疾病患病率
        
        for threshold in thresholds:
            # 特殊情况处理：阈值为1.0时
            if threshold == 1.0:
                net_benefit = 0
            else:
                # 修改后的代码：始终计算净收益，不管阈值是否大于患病率
                net_benefit = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
            
            # 不对"All"曲线进行截断，让绘图函数处理截断
            net_benefits.append(net_benefit)
        
        return net_benefits
    
    def plot_decision_curves(self, models_base_path):
        """
        绘制决策曲线
        
        参数:
        models_base_path: 模型基础路径
        """
        # 定义模型显示名称映射
        display_names = {
            'voting_ensemble': 'Voting',
            'stacking_model': 'Stacking'
        }
        
        X, y, sample_names = self.load_data()
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
        
        # 定义阈值范围，包含0.0和1.0这两个端点
        thresholds = np.arange(0.0, 1.01, 0.001)  # 步长为0.001，范围从0.0到1.0
        
        # 为每个模型创建存储净收益的字典
        model_benefits = {model_name: [] for model_name in self.model_names}
        
        # 存储每个折叠的真实标签
        fold_y_trues = []
        
        # 移除特殊模型处理逻辑，所有模型使用相同的加载方式
        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            print(f"处理第 {fold} 折...")
            
            # 创建当前折的详细记录
            fold_records = []
            
            # 记录训练集数据
            for idx in train_index:
                fold_records.append({
                    'filename': sample_names[idx],
                    'dataset_type': 'train',
                    'true_label': y[idx]
                })
            
            # 准备测试数据
            X_test = X.iloc[val_index]
            y_test = y[val_index]
            fold_y_trues.append(y_test)
            
            # 为每个模型计算净收益
            fold_benefits = {}  # 存储当前折叠下每个模型的净收益
            
            # 处理所有模型，包括常规模型和集成模型
            for model_name in self.model_names:
                print(f"  处理模型 {model_name}...")
                
                # 加载模型
                model_path = os.path.join(models_base_path, model_name, f'{model_name}_fold_{fold}.pkl')
                model_dict = joblib.load(model_path)
                model = model_dict["model"]
                scaler = model_dict["scaler"]
                
                # 数据标准化
                X_test_scaled = scaler.transform(X_test)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
                
                # 获取预测概率
                if hasattr(model, "predict_proba"):
                    y_scores = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    # 对于不支持predict_proba的模型，使用预测值作为近似
                    y_scores = model.predict(X_test_scaled)
                
                # 记录测试集数据和预测结果
                for idx, test_idx in enumerate(val_index):
                    # 如果是第一个模型，添加新记录
                    if model_name == self.model_names[0]:
                        fold_records.append({
                            'filename': sample_names[test_idx],
                            'dataset_type': 'test',
                            'true_label': y[test_idx],
                            f'{model_name}_prob': y_scores[idx]
                        })
                    # 如果不是第一个模型，更新现有记录
                    else:
                        fold_records[len(train_index) + idx][f'{model_name}_prob'] = y_scores[idx]
                
                # 计算净收益
                net_benefits = self.calculate_model_net_benefit(y_test, y_scores, thresholds)
                model_benefits[model_name].append(net_benefits)
                fold_benefits[model_name] = net_benefits
            
            # 为当前折叠计算"全部治疗"策略的净收益
            fold_all_positive_benefits = self.calculate_all_positive_net_benefit(y_test, thresholds)
            fold_none_benefits = [0] * len(thresholds)
            
            # 保存当前折的详细记录
            fold_df = pd.DataFrame(fold_records)
            
            # 创建fold目录
            fold_dir = os.path.join(self.save_path, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # 保存详细记录
            fold_df.to_csv(os.path.join(fold_dir, 'detailed_records.csv'), index=False)
            
            # 绘制当前折的决策曲线
            plt.figure(figsize=(12, 8), dpi=600)
            plt.rcParams['font.family'] = 'Times New Roman'
            
            # 绘制基准线
            plt.plot(thresholds, fold_none_benefits, 'k--', label='None', linewidth=2)
            plt.plot(thresholds, fold_all_positive_benefits, 'k-', label='All', linewidth=2)
            
            # 绘制每个模型的净收益曲线
            # 使用蓝绿青紫色系的配色方案
            blue_teal_purple_colors = [
                '#1f77b4',  # 蓝色
                '#4682B4',  # 钢蓝色
                '#00CED1',  # 深青色
                '#40E0D0',  # 绿松石色
                '#00FFFF',  # 青色
                '#00FA9A',  # 中春绿色
                '#48D1CC',  # 中绿松石色
                '#20B2AA',  # 浅海绿色
                '#5F9EA0',  # 军校蓝
                '#6495ED',  # 矢车菊蓝
                '#7B68EE',  # 中板岩蓝
                '#6A5ACD',  # 板岩蓝
                '#9370DB',  # 中紫色
                '#8A2BE2',  # 紫罗兰色
                '#9932CC'   # 深兰花紫
            ]
            
            # 使用全部实线
            for i, model_name in enumerate(self.model_names):
                # 使用显示名称而不是内部名称
                display_name = display_names.get(model_name, model_name)
                
                plt.plot(thresholds, fold_benefits[model_name], 
                         label=display_name, 
                         linewidth=2, 
                         color=blue_teal_purple_colors[i % len(blue_teal_purple_colors)])
            
            # 设置图表属性
            plt.xlabel('Threshold Probability', fontsize=14, fontname='Times New Roman')
            plt.ylabel('Net Benefit', fontsize=14, fontname='Times New Roman')
            plt.title(f'Decision Curve Analysis (Fold {fold})', fontsize=28, fontname='Times New Roman')
            plt.legend(fontsize=12, loc='best')
            plt.grid(True, alpha=0.3)
            
            # 设置坐标轴范围
            plt.xlim([0, 1])
            # 将纵轴范围固定为-0.1到0.5，但不截断曲线
            plt.ylim([-0.1, 0.5])
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(os.path.join(fold_dir, 'decision_curves.png'))
            plt.savefig(os.path.join(fold_dir, 'decision_curves.svg'), format='svg')
            plt.close()
            
            # 保存当前折的决策曲线数据
            fold_results = {
                'thresholds': thresholds,
                'all': fold_all_positive_benefits,
                'none': fold_none_benefits
            }
            for model_name in self.model_names:
                fold_results[model_name] = fold_benefits[model_name]
            
            fold_results_df = pd.DataFrame(fold_results)
            fold_results_df.to_csv(os.path.join(fold_dir, 'decision_curves_data.csv'), index=False)
        
        # 合并所有折叠的y_true
        all_y_true = np.concatenate(fold_y_trues)
        
        # 计算"全部治疗"和"无人治疗"策略的净收益
        all_positive_benefits = self.calculate_all_positive_net_benefit(all_y_true, thresholds)
        none_benefits = [0] * len(thresholds)  # "无人治疗"策略的净收益始终为0
        
        # 计算每个模型的平均净收益
        mean_benefits = {}
        for model_name in self.model_names:
            mean_benefits[model_name] = np.mean(model_benefits[model_name], axis=0)
        
        # 绘制决策曲线
        plt.figure(figsize=(12, 8), dpi=600)
        plt.rcParams['font.family'] = 'Times New Roman'
        
        # 绘制基准线
        plt.plot(thresholds, none_benefits, 'k--', label='None', linewidth=2)
        plt.plot(thresholds, all_positive_benefits, 'k-', label='All', linewidth=2)
        
        # 绘制每个模型的净收益曲线
        # 使用蓝绿青紫色系的配色方案
        blue_teal_purple_colors = [
            '#1f77b4',  # 蓝色
            '#4682B4',  # 钢蓝色
            '#00CED1',  # 深青色
            '#40E0D0',  # 绿松石色
            '#00FFFF',  # 青色
            '#00FA9A',  # 中春绿色
            '#48D1CC',  # 中绿松石色
            '#20B2AA',  # 浅海绿色
            '#5F9EA0',  # 军校蓝
            '#6495ED',  # 矢车菊蓝
            '#7B68EE',  # 中板岩蓝
            '#6A5ACD',  # 板岩蓝
            '#9370DB',  # 中紫色
            '#8A2BE2',  # 紫罗兰色
            '#9932CC'   # 深兰花紫
        ]
        
        # 使用全部实线
        for i, model_name in enumerate(self.model_names):
            # 使用显示名称而不是内部名称
            display_name = display_names.get(model_name, model_name)
            
            plt.plot(thresholds, mean_benefits[model_name], 
                     label=display_name, 
                     linewidth=2, 
                     color=blue_teal_purple_colors[i % len(blue_teal_purple_colors)])
        
        # 设置图表属性
        plt.xlabel('Threshold Probability', fontsize=14, fontname='Times New Roman')
        plt.ylabel('Net Benefit', fontsize=14, fontname='Times New Roman')
        plt.title('Decision Curve Analysis', fontsize=28, fontname='Times New Roman')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        
        # 设置坐标轴范围
        plt.xlim([0, 1])
        # 将平均图的纵轴范围也固定为-0.1到0.5
        plt.ylim([-0.1, 0.5])
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'decision_curves.png'))
        plt.savefig(os.path.join(self.save_path, 'decision_curves.svg'), format='svg')
        plt.close()
        
        # 保存数据
        results = {
            'thresholds': thresholds,
            'all': all_positive_benefits,
            'none': none_benefits
        }
        for model_name in self.model_names:
            results[model_name] = mean_benefits[model_name]
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.save_path, 'decision_curves_data.csv'), index=False)

    def compare_aug_nonaug_models(self, aug_path, nonaug_path, compare_save_path):
        """
        比较增强(aug)和非增强(non-aug)模型的决策曲线
        
        参数:
        aug_path: 增强模型的基础路径
        nonaug_path: 非增强模型的基础路径
        compare_save_path: 比较结果保存路径
        """
        print("开始比较增强和非增强模型的决策曲线...")
        os.makedirs(compare_save_path, exist_ok=True)
        
        # 定义模型显示名称映射
        display_names = {
            'voting_ensemble': 'Voting',
            'stacking_model': 'Stacking'
        }
        
        X, y, sample_names = self.load_data()
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
        
        # 定义阈值范围，包含0.0和1.0这两个端点
        thresholds = np.arange(0.0, 1.01, 0.001)
        
        # 为每个模型在aug和non-aug两种情况下创建净收益字典
        aug_fold_benefits = {model_name: {} for model_name in self.model_names}
        nonaug_fold_benefits = {model_name: {} for model_name in self.model_names}
        
        # 存储每个折叠的基准线数据
        fold_baselines = {}
        
        # 遍历每个折叠
        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            print(f"处理第 {fold} 折...")
            
            # 准备测试数据
            X_test = X.iloc[val_index]
            y_test = y[val_index]
            
            # 为当前折叠创建目录
            fold_dir = os.path.join(compare_save_path, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # 计算基准线
            fold_all_positive_benefits = self.calculate_all_positive_net_benefit(y_test, thresholds)
            fold_none_benefits = [0] * len(thresholds)
            
            # 存储基准线数据
            fold_baselines[fold] = {
                'all': fold_all_positive_benefits,
                'none': fold_none_benefits
            }
            
            # 为每个模型计算aug和non-aug两种情况下的净收益
            for model_name in self.model_names:
                print(f"  处理模型 {model_name}...")
                aug_fold_benefits[model_name][fold] = None
                nonaug_fold_benefits[model_name][fold] = None
                
                # 加载aug模型
                try:
                    aug_model_path = os.path.join(aug_path, model_name, f'{model_name}_fold_{fold}.pkl')
                    aug_model_dict = joblib.load(aug_model_path)
                    aug_model = aug_model_dict["model"]
                    aug_scaler = aug_model_dict["scaler"]
                    
                    # 数据标准化
                    X_test_scaled_aug = aug_scaler.transform(X_test)
                    X_test_scaled_aug = pd.DataFrame(X_test_scaled_aug, columns=X.columns)
                    
                    # 获取预测概率
                    if hasattr(aug_model, "predict_proba"):
                        aug_scores = aug_model.predict_proba(X_test_scaled_aug)[:, 1]
                    else:
                        aug_scores = aug_model.predict(X_test_scaled_aug)
                    
                    # 计算净收益
                    aug_net_benefits = self.calculate_model_net_benefit(y_test, aug_scores, thresholds)
                    aug_fold_benefits[model_name][fold] = aug_net_benefits
                except Exception as e:
                    print(f"加载aug模型 {model_name} 时出错: {e}")
                
                # 加载non-aug模型
                try:
                    nonaug_model_path = os.path.join(nonaug_path, model_name, f'{model_name}_fold_{fold}.pkl')
                    nonaug_model_dict = joblib.load(nonaug_model_path)
                    nonaug_model = nonaug_model_dict["model"]
                    nonaug_scaler = nonaug_model_dict["scaler"]
                    
                    # 数据标准化
                    X_test_scaled_nonaug = nonaug_scaler.transform(X_test)
                    X_test_scaled_nonaug = pd.DataFrame(X_test_scaled_nonaug, columns=X.columns)
                    
                    # 获取预测概率
                    if hasattr(nonaug_model, "predict_proba"):
                        nonaug_scores = nonaug_model.predict_proba(X_test_scaled_nonaug)[:, 1]
                    else:
                        nonaug_scores = nonaug_model.predict(X_test_scaled_nonaug)
                    
                    # 计算净收益
                    nonaug_net_benefits = self.calculate_model_net_benefit(y_test, nonaug_scores, thresholds)
                    nonaug_fold_benefits[model_name][fold] = nonaug_net_benefits
                except Exception as e:
                    print(f"加载non-aug模型 {model_name} 时出错: {e}")
                
                # 为当前模型和当前折叠生成比较图
                if aug_fold_benefits[model_name][fold] is not None or nonaug_fold_benefits[model_name][fold] is not None:
                    plt.figure(figsize=(12, 8), dpi=600)
                    plt.rcParams['font.family'] = 'Times New Roman'
                    
                    # 使用显示名称而不是内部名称
                    display_name = display_names.get(model_name, model_name)
                    
                    # 绘制基准线
                    plt.plot(thresholds, fold_none_benefits, 'k--', label='None', linewidth=2)
                    plt.plot(thresholds, fold_all_positive_benefits, 'k-', label='All', linewidth=2)
                    
                    # 绘制aug和non-aug模型的曲线
                    if aug_fold_benefits[model_name][fold] is not None:
                        plt.plot(thresholds, aug_fold_benefits[model_name][fold], 
                                label=f'{display_name} (VAE)', 
                                linewidth=2,
                                color='#1f77b4')  # 蓝色
                    
                    if nonaug_fold_benefits[model_name][fold] is not None:
                        plt.plot(thresholds, nonaug_fold_benefits[model_name][fold], 
                                label=f'{display_name} (Original)', 
                                linewidth=2,
                                color='#ff7f0e',  # 橙色
                                linestyle='--')  # 为非增强模型使用虚线以区分
                    
                    # 设置图表属性
                    plt.xlabel('Threshold Probability', fontsize=14, fontname='Times New Roman')
                    plt.ylabel('Net Benefit', fontsize=14, fontname='Times New Roman')
                    plt.title(f'Decision Curve Analysis - {display_name} (Fold {fold}) ', fontsize=28, fontname='Times New Roman')
                    plt.legend(fontsize=12, loc='best')
                    plt.grid(True, alpha=0.3)
                    
                    # 设置坐标轴范围
                    plt.xlim([0, 1])
                    plt.ylim([-0.1, 0.5])
                    
                    # 创建模型在当前折叠的目录
                    model_fold_dir = os.path.join(fold_dir, model_name)
                    os.makedirs(model_fold_dir, exist_ok=True)
                    
                    # 保存图表
                    plt.tight_layout()
                    plt.savefig(os.path.join(model_fold_dir, f'{model_name}_fold_{fold}_comparison.png'))
                    plt.savefig(os.path.join(model_fold_dir, f'{model_name}_fold_{fold}_comparison.svg'), format='svg')
                    plt.close()
                    
                    # 保存比较数据
                    comparison_data = {
                        'thresholds': thresholds,
                        'all': fold_all_positive_benefits,
                        'none': fold_none_benefits
                    }
                    
                    if aug_fold_benefits[model_name][fold] is not None:
                        comparison_data[f'{model_name}_aug'] = aug_fold_benefits[model_name][fold]
                    
                    if nonaug_fold_benefits[model_name][fold] is not None:
                        comparison_data[f'{model_name}_nonaug'] = nonaug_fold_benefits[model_name][fold]
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df.to_csv(os.path.join(model_fold_dir, f'{model_name}_fold_{fold}_comparison_data.csv'), index=False)
        
        # 计算每个模型在aug和non-aug两种情况下的平均净收益
        aug_mean_benefits = {}
        nonaug_mean_benefits = {}
        
        # 计算每个模型在aug和non-aug两种情况下的95%置信区间
        aug_ci_lower = {}
        aug_ci_upper = {}
        nonaug_ci_lower = {}
        nonaug_ci_upper = {}
        
        for model_name in self.model_names:
            # 处理aug模型
            valid_aug_benefits = [benefits for fold, benefits in aug_fold_benefits[model_name].items() if benefits is not None]
            if valid_aug_benefits:
                aug_mean_benefits[model_name] = np.mean(valid_aug_benefits, axis=0)
                
                # 计算95%置信区间
                if len(valid_aug_benefits) > 1:  # 至少有2个折叠才能计算置信区间
                    # 计算标准误差
                    aug_std = np.std(valid_aug_benefits, axis=0)
                    aug_se = aug_std / np.sqrt(len(valid_aug_benefits))
                    # 计算95%置信区间 (1.96 是 95% 置信水平的 Z 值)
                    aug_ci_lower[model_name] = aug_mean_benefits[model_name] - 1.96 * aug_se
                    aug_ci_upper[model_name] = aug_mean_benefits[model_name] + 1.96 * aug_se
                else:
                    aug_ci_lower[model_name] = None
                    aug_ci_upper[model_name] = None
            else:
                aug_mean_benefits[model_name] = None
                aug_ci_lower[model_name] = None
                aug_ci_upper[model_name] = None
            
            # 处理non-aug模型
            valid_nonaug_benefits = [benefits for fold, benefits in nonaug_fold_benefits[model_name].items() if benefits is not None]
            if valid_nonaug_benefits:
                nonaug_mean_benefits[model_name] = np.mean(valid_nonaug_benefits, axis=0)
                
                # 计算95%置信区间
                if len(valid_nonaug_benefits) > 1:  # 至少有2个折叠才能计算置信区间
                    # 计算标准误差
                    nonaug_std = np.std(valid_nonaug_benefits, axis=0)
                    nonaug_se = nonaug_std / np.sqrt(len(valid_nonaug_benefits))
                    # 计算95%置信区间
                    nonaug_ci_lower[model_name] = nonaug_mean_benefits[model_name] - 1.96 * nonaug_se
                    nonaug_ci_upper[model_name] = nonaug_mean_benefits[model_name] + 1.96 * nonaug_se
                else:
                    nonaug_ci_lower[model_name] = None
                    nonaug_ci_upper[model_name] = None
            else:
                nonaug_mean_benefits[model_name] = None
                nonaug_ci_lower[model_name] = None
                nonaug_ci_upper[model_name] = None
        
        # 计算平均基准线
        all_fold_all_benefits = [data['all'] for fold, data in fold_baselines.items()]
        mean_all_positive_benefits = np.mean(all_fold_all_benefits, axis=0)
        none_benefits = [0] * len(thresholds)
        
        # 创建平均结果目录
        avg_dir = os.path.join(compare_save_path, 'average')
        os.makedirs(avg_dir, exist_ok=True)
        
        # 为每个模型生成平均比较图
        for model_name in self.model_names:
            if aug_mean_benefits[model_name] is None and nonaug_mean_benefits[model_name] is None:
                print(f"模型 {model_name} 的aug和non-aug版本都不可用，跳过平均绘图")
                continue
            
            # 使用显示名称而不是内部名称
            display_name = display_names.get(model_name, model_name)
                
            plt.figure(figsize=(12, 8), dpi=600)
            plt.rcParams['font.family'] = 'Times New Roman'
            
            # 绘制基准线
            plt.plot(thresholds, none_benefits, 'k--', label='None', linewidth=2)
            plt.plot(thresholds, mean_all_positive_benefits, 'k-', label='All', linewidth=2)
            
            # 绘制aug和non-aug模型的曲线及其95%置信区间
            if aug_mean_benefits[model_name] is not None:
                plt.plot(thresholds, aug_mean_benefits[model_name], 
                        label=f'{display_name} (VAE)', 
                        linewidth=2,
                        color='#1f77b4')  # 蓝色
                
                # 绘制95%置信区间
                if aug_ci_lower[model_name] is not None and aug_ci_upper[model_name] is not None:
                    plt.fill_between(thresholds, 
                                    aug_ci_lower[model_name], 
                                    aug_ci_upper[model_name], 
                                    color='#1f77b4', 
                                    alpha=0.2,
                                    label=f'{display_name} (VAE) 95% CI')
            
            if nonaug_mean_benefits[model_name] is not None:
                plt.plot(thresholds, nonaug_mean_benefits[model_name], 
                        label=f'{display_name} (Original)', 
                        linewidth=2,
                        color='#ff7f0e',  # 橙色
                        linestyle='--')  # 为非增强模型使用虚线以区分
                
                # 绘制95%置信区间
                if nonaug_ci_lower[model_name] is not None and nonaug_ci_upper[model_name] is not None:
                    plt.fill_between(thresholds, 
                                    nonaug_ci_lower[model_name], 
                                    nonaug_ci_upper[model_name], 
                                    color='#ff7f0e', 
                                    alpha=0.2,
                                    label=f'{display_name} (Original) 95% CI')
            
            # 设置图表属性
            plt.xlabel('Threshold Probability', fontsize=14, fontname='Times New Roman')
            plt.ylabel('Net Benefit', fontsize=14, fontname='Times New Roman')
            plt.title(f'Decision Curve Analysis - {display_name} (Average)', fontsize=28, fontname='Times New Roman')
            plt.legend(fontsize=12, loc='best')
            plt.grid(True, alpha=0.3)
            
            # 设置坐标轴范围
            plt.xlim([0, 1])
            plt.ylim([-0.1, 0.5])
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(os.path.join(avg_dir, f'{model_name}_avg_comparison.png'))
            plt.savefig(os.path.join(avg_dir, f'{model_name}_avg_comparison.svg'), format='svg')
            plt.close()
            
            # 保存比较数据
            comparison_data = {
                'thresholds': thresholds,
                'all': mean_all_positive_benefits,
                'none': none_benefits
            }
            
            if aug_mean_benefits[model_name] is not None:
                comparison_data[f'{model_name}_aug'] = aug_mean_benefits[model_name]
                if aug_ci_lower[model_name] is not None:
                    comparison_data[f'{model_name}_aug_ci_lower'] = aug_ci_lower[model_name]
                    comparison_data[f'{model_name}_aug_ci_upper'] = aug_ci_upper[model_name]
            
            if nonaug_mean_benefits[model_name] is not None:
                comparison_data[f'{model_name}_nonaug'] = nonaug_mean_benefits[model_name]
                if nonaug_ci_lower[model_name] is not None:
                    comparison_data[f'{model_name}_nonaug_ci_lower'] = nonaug_ci_lower[model_name]
                    comparison_data[f'{model_name}_nonaug_ci_upper'] = nonaug_ci_upper[model_name]
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(os.path.join(avg_dir, f'{model_name}_avg_comparison_data.csv'), index=False)
        
        print("aug和non-aug模型的决策曲线比较完成！结果已保存到:", compare_save_path)

    def create_combined_comparison_plot(self, aug_path, nonaug_path, compare_save_path):
        """
        创建组合图表，将所有模型的平均决策曲线比较图放在一张图中
        
        参数:
        aug_path: 增强模型的基础路径
        nonaug_path: 非增强模型的基础路径
        compare_save_path: 比较结果保存路径
        """
        print("开始创建组合决策曲线比较图...")
        
        # 首先运行常规的比较函数以确保所有数据都已计算
        self.compare_aug_nonaug_models(aug_path, nonaug_path, compare_save_path)
        
        # 定义模型显示顺序
        model_display_order = [
            'voting_ensemble',  # Voting
            'stacking_model',   # Stacking
            'RF',               # RF
            'LightGBM',         # LightGBM
            'XGBoost',          # XGBoost
            'GBDT',             # GBDT
            'LogisticRegression', # LogisticRegression
            'SVM',              # SVM
            'AdaBoost',         # AdaBoost
            'MLP'               # MLP
        ]
        
        # 定义模型显示名称映射
        display_names = {
            'voting_ensemble': 'Voting',
            'stacking_model': 'Stacking'
        }
        
        # 创建子图网格 (3行4列，最后一行只有2个)
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15), dpi=300)
        plt.rcParams['font.family'] = 'Times New Roman'
        
        # 加载所有模型的数据
        avg_dir = os.path.join(compare_save_path, 'average')
        
        # 展平axes数组以便访问
        axes_flat = axes.flatten()
        
        # 将右下角子图用于图例
        legend_ax = axes_flat[11]
        # 隐藏倒数第二个子图
        axes_flat[10].set_visible(False)
        
        # 清除图例子图的坐标轴
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])
        legend_ax.axis('off')
        
        # 创建图例元素
        legend_elements = [
            plt.Line2D([0], [0], color='k', linestyle='--', label='None', linewidth=2),
            plt.Line2D([0], [0], color='k', linestyle='-', label='All', linewidth=2),
            plt.Line2D([0], [0], color='#1f77b4', linestyle='-', label='VAE', linewidth=2.5),
            plt.Line2D([0], [0], color='#ff7f0e', linestyle='--', label='Original', linewidth=2.5),
            plt.Rectangle((0, 0), 1, 1, fc='#1f77b4', alpha=0.2, label='VAE 95% CI'),
            plt.Rectangle((0, 0), 1, 1, fc='#ff7f0e', alpha=0.2, label='Original 95% CI')
        ]
        
        # 添加图例到右下角子图
        legend_ax.legend(handles=legend_elements, loc='center', fontsize=16)
        
        # 遍历模型顺序，绘制每个模型的决策曲线
        for idx, model_name in enumerate(model_display_order):
            if idx >= 10:  # 跳过超出9个模型的部分
                continue
                
            ax = axes_flat[idx]
            
            # 加载该模型的比较数据
            data_file = os.path.join(avg_dir, f'{model_name}_avg_comparison_data.csv')
            
            try:
                df = pd.read_csv(data_file)
                
                # 获取数据
                thresholds = df['thresholds'].values
                all_benefits = df['all'].values
                none_benefits = df['none'].values
                
                # 使用显示名称而不是内部名称
                display_name = display_names.get(model_name, model_name)
                
                # 绘制基准线
                ax.plot(thresholds, none_benefits, 'k--', label='None', linewidth=2)
                ax.plot(thresholds, all_benefits, 'k-', label='All', linewidth=2)
                
                # 绘制模型曲线
                if f'{model_name}_aug' in df.columns:
                    aug_benefits = df[f'{model_name}_aug'].values
                    ax.plot(thresholds, aug_benefits, 
                            label='VAE', 
                            linewidth=2.5,
                            color='#1f77b4')  # 蓝色
                    
                    # 绘制95%置信区间
                    if f'{model_name}_aug_ci_lower' in df.columns and f'{model_name}_aug_ci_upper' in df.columns:
                        aug_ci_lower = df[f'{model_name}_aug_ci_lower'].values
                        aug_ci_upper = df[f'{model_name}_aug_ci_upper'].values
                        ax.fill_between(thresholds, 
                                      aug_ci_lower, 
                                      aug_ci_upper, 
                                      color='#1f77b4', 
                                      alpha=0.2)
                
                if f'{model_name}_nonaug' in df.columns:
                    nonaug_benefits = df[f'{model_name}_nonaug'].values
                    ax.plot(thresholds, nonaug_benefits, 
                            label='Original', 
                            linewidth=2.5,
                            color='#ff7f0e',  # 橙色
                            linestyle='--')
                    
                    # 绘制95%置信区间
                    if f'{model_name}_nonaug_ci_lower' in df.columns and f'{model_name}_nonaug_ci_upper' in df.columns:
                        nonaug_ci_lower = df[f'{model_name}_nonaug_ci_lower'].values
                        nonaug_ci_upper = df[f'{model_name}_nonaug_ci_upper'].values
                        ax.fill_between(thresholds, 
                                      nonaug_ci_lower, 
                                      nonaug_ci_upper, 
                                      color='#ff7f0e', 
                                      alpha=0.2)
                
                # 设置坐标轴范围
                ax.set_xlim([0, 1])
                ax.set_ylim([-0.1, 0.5])
                
                # 设置图表标题和坐标轴标签
                ax.set_title(display_name, fontsize=18, fontname='Times New Roman')
                
                # 调整X轴标签显示逻辑 - 只在第二行最后两个和第三行所有显示的子图上显示
                # 第二行最后两个子图索引为6,7，第三行前两个索引为8,9
                if idx >= 6:  # 第二行后两个和第三行的
                    ax.set_xlabel('Threshold Probability', fontsize=15, fontname='Times New Roman')
                else:
                    ax.set_xticklabels([])  # 其他子图不显示X轴标签
                
                # 只给最左列的子图添加Y轴标签
                if idx % 4 == 0:  # 每行第一个
                    ax.set_ylabel('Net Benefit', fontsize=15, fontname='Times New Roman')
                
                # 设置刻度标签字体大小
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                # 为每个子图添加网格线
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"处理模型 {model_name} 时出错: {e}")
                ax.text(0.5, 0.5, f"{display_name}\n(数据不可用)", 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=ax.transAxes, fontsize=14)
        
        # 调整子图间距
        plt.tight_layout()
        
        # 保存图表为高质量PDF
        combined_plot_path = os.path.join(compare_save_path, 'all_models_comparison.pdf')
        plt.savefig(combined_plot_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(compare_save_path, 'all_models_comparison.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"组合决策曲线比较图已保存到: {combined_plot_path}")

if __name__ == "__main__":
    # 设置路径
    data_paths = [
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_1.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_2.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_3.csv',
        r'D:\Desktop\parotid_XGBoost\data\new_data\label_4.csv'
    ]
    
    # 输出路径
    # save_path = r'DCA\origin'
    save_path = r'DCA\aug'
    
    # 模型基本路径
    # models_base_path = r'other_KFold_model'
    models_base_path = r'other_KFold_model_aug'
    
    # 要比较的模型列表
    model_names = ['XGBoost', 'LightGBM', 'LogisticRegression', 'AdaBoost', 
                   'GBDT', 'MLP', 'SVM', 'RF', 'voting_ensemble', 'stacking_model']
    
    # 创建分析器
    analyzer = DecisionCurveAnalysis(model_names, data_paths, save_path)
    
    # 选择要执行的功能
    # 1. 只分析一组模型
    # analyzer.plot_decision_curves(models_base_path)
    
    # 2. 比较aug和non-aug模型
    aug_path = r'other_KFold_model_aug'
    nonaug_path = r'other_KFold_model'
    compare_save_path = r'DCA\model_comparison'
    
    # 3. 创建组合比较图
    analyzer.create_combined_comparison_plot(aug_path, nonaug_path, compare_save_path)
    
    print("决策曲线分析完成！") 