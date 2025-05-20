import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import joblib
import shap
import matplotlib.pyplot as plt
import gc
import base64
from io import BytesIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch
import nibabel as nib
import numpy as np
from scipy import ndimage

# 在项目目录下创建所需文件夹
project_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['nnUNet_raw'] = os.path.join(project_dir, 'nnunet_raw')
os.environ['nnUNet_results'] = os.path.join(project_dir, 'nnunet_results')

# 确保目录存在
os.makedirs(os.environ['nnUNet_raw'], exist_ok=True)
os.makedirs(os.environ['nnUNet_results'], exist_ok=True)


class RadiomicsAnalyzer:
    def __init__(self):
        # 初始化特征提取器
        self._settings = {'geometryTolerance': 10}
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**self._settings)
        
        # 配置特征提取器
        self._configure_extractor()
        
        # 加载预测模型
        model_path = os.path.join('RF_fold_3.pkl')
        model_dict = joblib.load(model_path)
        self.rf_model = model_dict["model"]
        self.scaler = model_dict["scaler"]

    def _configure_extractor(self):
        """配置特征提取器的设置"""
        _settings = {
            'geometryTolerance': 10,
            'enableDiagnostics': False,
            'normalize': False,
            'removeOutliers': None,
            'force2D': False,
            'binWidth': 25,
            'voxelArrayShift': 0,
            
            # 小波变换设置
            'wavelet': 'coif1',
            'start_level': 0,
            'level': 1,
            
            # LBP设置
            'lbp2DRadius': 1,
            'lbp2DPoints': 8,
            
            # 梯度设置
            'gradientUseSpacing': True
        }
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**_settings)
        
        # 启用所有基本特征类型
        self.extractor.enableFeatureClassByName('shape')      # 14个特征
        self.extractor.enableFeatureClassByName('firstorder') # 18个特征
        self.extractor.enableFeatureClassByName('glcm')      # 24个特征
        self.extractor.enableFeatureClassByName('glrlm')     # 16个特征
        self.extractor.enableFeatureClassByName('glszm')     # 16个特征
        self.extractor.enableFeatureClassByName('gldm')      # 14个特征
        self.extractor.enableFeatureClassByName('ngtdm')     # 5个特征
        
        # 启用所需的图像类型
        self.extractor.enableImageTypeByName('Original')     # 原始图像
        self.extractor.enableImageTypeByName('Exponential')  # 指数变换
        self.extractor.enableImageTypeByName('Logarithm')    # 对数变换
        self.extractor.enableImageTypeByName('Square')       # 平方变换
        self.extractor.enableImageTypeByName('SquareRoot')   # 平方根变换
        self.extractor.enableImageTypeByName('Gradient')     # 梯度滤波
        self.extractor.enableImageTypeByName('LBP2D')       # 局部二值模式
        self.extractor.enableImageTypeByName('Wavelet')      # 小波变换

    def extract_features(self, save_time, image_num=1):
        """提取放射组学特征"""
        # 设置文件路径
        image_file = os.path.join(r"upload_files/images", f'image_{save_time}.nii.gz')
        label_file = os.path.join(r"upload_files/labels", f'label_{save_time}.nii.gz')
        
        # 设置CSV文件路径
        csv_files = {
            1: f"radiomics_features/{image_num}/{save_time}_label_1.csv",
            # 2: f"radiomics_features/{save_time}_label_2.csv",
            3: f"radiomics_features/{image_num}/{save_time}_label_2.csv",
            # 4: f"radiomics_features/{save_time}_label_4.csv"
        }

        # 确保radiomics_features和子目录存在
        os.makedirs(os.path.join("radiomics_features", image_num), exist_ok=True)

        # 读取图像
        image_nii = sitk.ReadImage(image_file)
        label_nii = sitk.ReadImage(label_file)

        # 为每个标签提取特征
        for label, csv_file in csv_files.items():
            self.extractor.settings['label'] = label
            feature = self.extractor.execute(image_nii, label_nii)
            result = pd.DataFrame([feature])
            result.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file))


    def predict(self, save_time, image_num=1):
        """进行预测并生成可视化"""
        # 读取特征文件
        feature_data = []
        labels = ['parotid_', 'submandibular_']
        
        for i, prefix in enumerate(labels, 1):
            file_path = f"radiomics_features/{image_num}/{save_time}_label_{i}.csv"
            data = pd.read_csv(file_path, index_col=0)
            X = data.iloc[:, 21:]  # 跳过前21列
            X.columns = [prefix + col for col in X.columns]
            feature_data.append(X)

        # 合并特征
        X = pd.concat([feature_data[0], feature_data[1]], axis=1)
        X = X[self.rf_model.feature_names_in_]
        feature_names = X.columns.tolist()
        
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            print(f"\n缩放错误: {str(e)}")
            print(f"错误类型: {type(e)}")
            raise
        
        # 修改SHAP配置，使用raw输出
        explainer = shap.TreeExplainer(
            self.rf_model,
            feature_perturbation='tree_path_dependent',
            model_output='raw'
        )
        
        # 获取SHAP值时需要传入原始数据
        explanation = explainer(X_scaled)
        
        # 统一处理不同版本的SHAP API
        try:
            # 新版本SHAP
            base_value = explainer.expected_value[1]  # 获取正类的基准值
            shap_values = explanation.values[..., 1]  # 获取正类的SHAP值
        except AttributeError:
            # 旧版本SHAP
            shap_values = explainer.shap_values(X_scaled)
            base_value = explainer.expected_value
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 获取正类的SHAP值
                base_value = base_value[1]
        
        # 生成SHAP解释图时传递原始特征值
        save_dir = os.path.join("shap_force_plot", image_num)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"shap_plot_{save_time}.pdf")
        graph = self._plot_shap_force(X, save_path)  # 改为传递原始特征DataFrame
        
        # 预测
        y_scores = self.rf_model.predict_proba(X_scaled)[:, 1]
        
        # 先创建所有特征的映射
        all_features = {
            self.rf_model.feature_names_in_[i]: {
                'name': self.rf_model.feature_names_in_[i],
                'value': float(shap_values[0][i]),  # 获取第一个样本的SHAP值
                'plot_index': i,
                'abs_value': abs(float(shap_values[0][i]))
            }
            for i in range(len(self.rf_model.feature_names_in_))
        }
        
        # 按绝对值大小排序并获取前5个
        top_5_features = dict(sorted(
            all_features.items(), 
            key=lambda x: x[1]['abs_value'], 
            reverse=True
        )[:5])
        
        # 创建最终的feature_mapping，保持原有的plot_index
        feature_mapping = {
            str(idx + 1): {
                'name': feature_info['name'],
                'value': feature_info['value'],
                'plot_index': feature_info['plot_index']
            }
            for idx, (_, feature_info) in enumerate(top_5_features.items())
        }
        
        # # 清理特征文件
        # for i in range(1, 5):
        #     try:
        #         feature_file = f"radiomics_features/{save_time}_label_{i}.csv"
        #         if os.path.exists(feature_file):
        #             os.remove(feature_file)
        #     except Exception as e:
        #         print(f"Warning: Failed to delete feature file {i}: {str(e)}")
        
        return y_scores.item(), graph, feature_mapping

    def _plot_shap_force(self, X, save_path):
        """生成SHAP力图"""
        explainer = shap.TreeExplainer(
            self.rf_model,
            feature_perturbation='tree_path_dependent',
            model_output='raw'
        )
        X_scaled = self.scaler.transform(X)
        
        # 获取SHAP值和基准值
        explanation = explainer(X_scaled)
        shap_values = explanation.values[..., 1]  # 获取正类的SHAP值
        base_value = explainer.expected_value[1]  # 获取正类的基准值
        
        # 创建图形
        plt.figure(figsize=(36, 16))
        
        try:
            shap.plots.force(
                base_value,
                shap_values[0],  # 使用第一个样本的SHAP值
                features=X.iloc[0],
                feature_names=self.rf_model.feature_names_in_,
                matplotlib=True,
                show=False
            )
            
            # 直接保存矢量格式
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存PDF（矢量格式）
                plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=400)
                
        except Exception as e:
            return None

        # # 获取影响最大的三个特征
        # feature_importance = np.abs(shap_values[0])
        # top_indices = np.argsort(feature_importance)[-3:][::-1]
        
        # # 在底部添加文本说明
        # text = "Top influential features:\n"
        # for i, idx in enumerate(top_indices, 1):
        #     feature_name = self.rf_model.feature_names_in_[idx]
        #     value = float(shap_values[0][idx])
        #     direction = "increases" if value > 0 else "decreases"
        #     text += f"{i}. {feature_name}: {direction} risk\n"
        
        # plt.figtext(0.05, 0.02, text, fontsize=16, ha='left', va='bottom')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        graph = base64.b64encode(image_png).decode('utf-8')
        gc.collect()
        return graph
    

def mask_segmentation(input_image_path, output_file_path, model_folder='mask_seg'):
    """进行掩膜分割"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        allow_tqdm=True
    )
    
    try:
        # 创建输入和输出目录
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # 确保输入文件存在
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Input file not found: {input_image_path}")
            
        # 初始化预测器
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=(0,),
            checkpoint_name='mask_seg.pth'
        )
        
        # 创建输入文件列表
        input_files = [[input_image_path]]  # 注意这里是双层列表
        
        # 使用predict_from_files方法进行预测
        predictor.predict_from_files(
            list_of_lists_or_source_folder=input_files,
            output_folder_or_list_of_truncated_output_files=output_file_path,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1
        )
        
        return True
    except Exception as e:
        print(f"Segmentation error: {str(e)}")
        return False


def process_labels(labels, target_label):
    """
    处理特定标签值的图像，保留最大的两个连通域，并根据重心位置进行映射。
    """
    # 创建一个与 labels 形状相同且填充为0的新数组
    target_mask = np.zeros_like(labels)

    # 将 labels 中所有等于 target_label 的位置复制到 target_mask 中
    target_mask[labels == target_label] = labels[labels == target_label]

    # 现在 target_mask 中只有 target_label 的值，其余位置为0
    labeled_array, num_features = ndimage.label(target_mask > 0)  # 使用 > 0 来确保只有非零值被标记

    if num_features < 2:
        return labels

    feature_info = {}
    for feature_id in range(1, num_features + 1):
        feature_size = np.sum(labeled_array == feature_id)
        center_of_mass = ndimage.center_of_mass(labeled_array == feature_id)
        feature_info[feature_id] = (feature_size, center_of_mass)

    sorted_features = sorted(feature_info.items(), key=lambda x: x[1][0], reverse=True)
    largest_two = sorted_features[:2]

    # 初始化结果数组
    result = labels.copy()

    # 保存两个最大连通域的特征ID
    feature_ids = [f_id for f_id, _ in largest_two]

    # 确定重心位置较小的连通域
    smaller_feature_id = min(largest_two, key=lambda x: x[1][1])  # 取元组中的第一个元素，即特征ID

    # 将重心位置较小的那个连通域设为target_label + 1
    mask = (labeled_array == smaller_feature_id[0])
    result = np.where(mask, target_label + 1, result)

    # 确保result中的非最大连通域部分为0，但仅限于原本是target_label的部分
    mask_target = (labels == target_label) & (~np.isin(labeled_array, feature_ids))
    result[mask_target] = 0

    return result

def process_mask(input_mask_path, output_mask_path, label_map):
    """处理单个掩膜文件，重新映射标签，保留最大的两个连通域，并根据重心位置进行映射。"""
    # 加载掩膜文件
    mask_nii = nib.load(input_mask_path)
    labels = np.round(mask_nii.get_fdata()).astype(np.int32)

    # 检查维度
    if len(labels.shape) != 3:
        print(f"Warning: Skipping file {input_mask_path} because it does not have 3 dimensions.")
        return

    # 标签重新映射
    modified_labels = np.zeros_like(labels)
    for old_label, new_label in label_map.items():
        modified_labels[labels == old_label] = new_label

    # 处理标签值为1的情况
    modified_labels = process_labels(modified_labels, 1)

    # 处理标签值为3的情况
    modified_labels = process_labels(modified_labels, 3)

    # 创建新的Nifti1Image对象
    new_img = nib.Nifti1Image(modified_labels, mask_nii.affine, header=mask_nii.header)

    # 保存修改后的数据
    nib.save(new_img, output_mask_path)
    print(f"Processed {input_mask_path}")


if __name__ == '__main__':
    # 设置输入和输出文件路径
    # input_mask_path = r'upload_files/masks/image.nii.gz'
    # output_mask_path = r'upload_files/labels/label_202411.nii.gz'
    # label_map = {40: 1, 41: 1, 38: 3, 39: 3}
    # process_mask(input_mask_path, output_mask_path, label_map)
    
    try:
        # 初始化分析器
        analyzer = RadiomicsAnalyzer()
        
        # 设置测试用例的时间戳
        test_time = "20250216"
        image_num = "348"
        # 126-3  287-2
        
        print("=== 开始放射组学分析 ===")
        print(f"时间戳: {test_time}")
        
        # 1. 提取特征
        print("\n1. 开始提取特征...")
        # analyzer.extract_features(test_time, image_num)
        print("特征提取完成")
        
        # 2. 进行预测
        print("\n2. 开始预测...")
        predict_result, graph, feature_mapping = analyzer.predict(test_time, image_num)
        predict_result = round(predict_result * 100, 2)
        
        # 3. 输出结果
        print("\n=== 分析结果 ===")
        print(f"预测概率: {predict_result}%")
        print("\n重要特征:")
        for rank, feature_info in feature_mapping.items():
            direction = "增加" if feature_info['value'] > 0 else "减少"
            print(f"No.{rank}: {feature_info['name']} - {direction}风险")
            
        # 4. 保存SHAP图
        if graph:
            # 创建保存目录
            save_dir = os.path.join("shap_force_plot", image_num)
            os.makedirs(save_dir, exist_ok=True)
            
            # 解码base64数据
            image_data = base64.b64decode(graph)
            
            # 保存不同格式
            for ext in ['svg', 'png']:
                shap_path = os.path.join(save_dir, f"shap_plot_{test_time}.{ext}")
                with open(shap_path, "wb") as f:
                    f.write(image_data)
            
            print(f"\nSHAP解释图已保存至: {save_dir}")
            
        print("\n=== 分析完成 ===")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
