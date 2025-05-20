import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
import logging
import os

class FeatureMapGenerator3D:
    def __init__(self):
        self.kernel_radius = 3
        
    def generate_feature_map(self, image_path, roi_path, feature_name, kernel_radius=3):
        """生成3D特征图
        
        Args:
            image_path: str, 输入图像路径
            roi_path: str, ROI掩码路径
            feature_name: str, 特征名称
            kernel_radius: int, 特征计算的核半径
            
        Returns:
            SimpleITK.Image: 生成的3D特征图
        """
        try:
            # 设置日志级别
            logging.getLogger('radiomics').setLevel(logging.ERROR)
            
            # 读取图像
            original_image = sitk.ReadImage(image_path)
            roi_image = sitk.ReadImage(roi_path)
            
            # 预处理ROI为二值掩码
            roi_array = sitk.GetArrayFromImage(roi_image)
            binary_roi = np.where(roi_array > 0, 1, 0).astype(np.uint8)
            binary_roi_image = sitk.GetImageFromArray(binary_roi)
            binary_roi_image.CopyInformation(roi_image)
            
            # 配置特征提取器
            setting_dict = {
                'binWidth': 25,
                'voxelBatch': 1000,
                'kernelRadius': kernel_radius,
                'correctMask': True,
                'force2D': False,  # 使用3D特征计算
                'label': 1,
            }
            
            extractor = featureextractor.RadiomicsFeatureExtractor(**setting_dict)
            
            # 禁用所有特征和图像类型
            extractor.disableAllFeatures()
            extractor.disableAllImageTypes()
            
            # 根据特征名称配置提取器
            if 'original' in feature_name:
                extractor.enableImageTypeByName('Original')
            elif 'square' in feature_name:
                extractor.enableImageTypeByName('Square')
            elif 'squareroot' in feature_name:
                extractor.enableImageTypeByName('SquareRoot')
            elif 'logarithm' in feature_name:
                extractor.enableImageTypeByName('Logarithm')

            # 根据特征类型设置特定参数
            if 'glcm' in feature_name:
                extractor.settings.update({
                    'force2D': True,
                    'force2Ddimension': 0,
                    'distances': [1],
                    'voxelBatch': 500,  # 减小批处理大小
                    'kernelRadius': min(kernel_radius, 3),  # 限制kernel大小
                    'angles': [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
                })
            elif 'glrlm' in feature_name or 'glszm' in feature_name:
                extractor.settings.update({
                    'force2D': True,
                    'force2Ddimension': 0,
                    'binWidth': 25
                })
            elif 'gldm' in feature_name:
                extractor.settings.update({
                    'force2D': True,
                    'force2Ddimension': 0,
                    'gldm_a': 0,
                    'binWidth': 25
                })

            # 修改Square预处理的设置
            if 'square' in feature_name:
                extractor.settings.update({
                    'normalize': False,
                    'removeOutliers': None,
                    'binWidth': 25,
                    'kernelRadius': 5,
                    'force2D': True,
                    'force2Ddimension': 0,
                    'distances': [1],
                    'angles': [0],
                    'voxelBatch': 1000,
                    'preCrop': True,
                    'label': 1,
                    'geometryTolerance': 0.0001,
                })

            # 启用相应的特征类别
            if 'firstorder' in feature_name:
                extractor.enableFeatureClassByName('firstorder')
                if 'Kurtosis' in feature_name:
                    extractor.enableFeaturesByName(firstorder=['Kurtosis'])

            elif 'glrlm' in feature_name:
                extractor.enableFeatureClassByName('glrlm')
                if 'LongRunHighGrayLevelEmphasis' in feature_name:
                    extractor.enableFeaturesByName(glrlm=['LongRunHighGrayLevelEmphasis'])
                elif 'RunEntropy' in feature_name:
                    extractor.enableFeaturesByName(glrlm=['RunEntropy'])
                elif 'ShortRunEmphasis' in feature_name:
                    extractor.enableFeaturesByName(glrlm=['ShortRunEmphasis'])
                elif 'RunPercentage' in feature_name:
                    extractor.enableFeaturesByName(glrlm=['RunPercentage'])

            elif 'gldm' in feature_name:
                extractor.enableFeatureClassByName('gldm')
                if 'DependenceNonUniformity' in feature_name:
                    extractor.enableFeaturesByName(gldm=['DependenceNonUniformity'])
                elif 'GrayLevelNonUniformity' in feature_name:
                    extractor.enableFeaturesByName(gldm=['GrayLevelNonUniformity'])

            elif 'glcm' in feature_name:
                extractor.enableFeatureClassByName('glcm')
                if 'ClusterShade' in feature_name:
                    extractor.enableFeaturesByName(glcm=['ClusterShade'])
                elif 'Contrast' in feature_name:
                    extractor.enableFeaturesByName(glcm=['Contrast'])
                elif 'DifferenceAverage' in feature_name:
                    extractor.enableFeaturesByName(glcm=['DifferenceAverage'])
                elif 'MaximumProbability' in feature_name:
                    extractor.enableFeaturesByName(glcm=['MaximumProbability'])

            elif 'glszm' in feature_name:
                extractor.enableFeatureClassByName('glszm')
                if 'GrayLevelVariance' in feature_name:
                    extractor.enableFeaturesByName(glszm=['GrayLevelVariance'])

            elif 'shape' in feature_name:
                extractor.enableFeatureClassByName('shape')
                if 'SurfaceArea' in feature_name:
                    extractor.enableFeaturesByName(shape=['SurfaceArea'])
            
            # 执行特征提取
            result = extractor.execute(original_image, binary_roi_image, voxelBased=True)
            
            # 获取特征图
            feature_map = None
            for key, val in result.items():
                if isinstance(val, sitk.Image):
                    feature_map = val
                    break
            
             # 重采样特征图以匹配原始图像
            resample = sitk.ResampleImageFilter()
            resample.SetReferenceImage(original_image)
            resample.SetInterpolator(sitk.sitkLinear)
            feature_map = resample.Execute(feature_map)
            
            if feature_map is None:
                raise ValueError("未找到特征图")
            
            # 处理特征图
            feature_array = sitk.GetArrayFromImage(feature_map)
            
            # 应用ROI掩码
            feature_array[binary_roi == 0] = 0
            
            # 移除归一化处理，保留原始特征值
            # valid_mask = binary_roi > 0
            # if valid_mask.any():
            #     valid_values = feature_array[valid_mask]
            #     if len(valid_values) > 0 and not np.all(valid_values == 0):
            #         min_val = np.min(valid_values)
            #         max_val = np.max(valid_values)
            #         if max_val > min_val:
            #             feature_array[valid_mask] = (feature_array[valid_mask] - min_val) / (max_val - min_val)
            
            # 创建最终特征图
            result_image = sitk.GetImageFromArray(feature_array)
            result_image.CopyInformation(original_image)
            
            return result_image
            
        except Exception as e:
            print(f"特征图生成失败: {str(e)}")
            return None

def main():
    # 测试代码
    num = 287
    image_path = rf'test_data\negative\image_SJS_0{num}.nii.gz'
    roi_path = rf'test_data\negative\label_SJS_0{num}.nii.gz'
    # image_path = rf'test_data\positive\image_SJS_0{num}.nii.gz'
    # roi_path = rf'test_data\positive\label_SJS_0{num}.nii.gz'
    base_name = os.path.basename(image_path)
    output_dir = os.path.join('3D_feature_maps', base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建特征图生成器
    generator = FeatureMapGenerator3D()
    
    # 测试不同特征
    feature_names = [
        'square_glrlm_ShortRunEmphasis',
        'square_glrlm_RunPercentage',
        'square_glcm_DifferenceAverage',
        'square_glcm_MaximumProbability'
    ]
    
    for feature_name in feature_names:
        print(f"\n处理特征: {feature_name}")
        
        # 生成特征图
        feature_map = generator.generate_feature_map(
            image_path=image_path,
            roi_path=roi_path,
            feature_name=feature_name,
            kernel_radius=3
        )
        
        if feature_map is not None:
            # 保存特征图
            output_path = os.path.join(output_dir, f"{feature_name}.nii.gz")
            sitk.WriteImage(feature_map, output_path)
            print(f"特征图已保存到: {output_path}")
        else:
            print(f"特征 {feature_name} 生成失败")

if __name__ == '__main__':
    main()
