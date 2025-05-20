import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from radiomics import featureextractor
import logging
from io import BytesIO
import base64
import gc
from OptimizedImageFilter import OptimizedImageFilter  # 引入OptimizedImageFilter类

class RadiomicsFeatureVisualizer:
    """放射组学特征可视化器"""
    
    def __init__(self):
        self.kernel_radius = 3  # 默认kernel半径
        self.filter_generator = OptimizedImageFilter()  # 创建滤波器实例
        self.reference_image = None  # 添加reference_image属性
        
    def generate_and_visualize_features(self, 
                                       image_path: str,
                                       roi_path: str,
                                       slice_positions: list,
                                       feature_names: list,
                                       kernel_radius: int = 3,
                                       save_to_file: bool = False,
                                       output_dir: str = None):
        """生成并可视化多个切片的多个放射组学特征图
        
        Args:
            image_path: 输入图像路径
            roi_path: ROI掩码路径
            slice_positions: 切片位置列表（每个元素为0-1之间的浮点数）
            feature_names: 特征名称列表
            kernel_radius: 特征计算的核半径
            save_to_file: 是否保存特征图到文件
            output_dir: 保存特征图的目录
            
        Returns:
            dict: 每个特征和位置的可视化结果（base64编码的PNG图像）
        """
        try:
            # 读取并保存参考图像
            self.reference_image = sitk.ReadImage(image_path)
            
            # 预处理ROI
            roi_image = self._preprocess_roi(roi_path, image_path)
            roi_array = sitk.GetArrayFromImage(roi_image)
            
            # 打印空间信息确认
            print("\n空间信息确认:")
            print(f"原始图像: Spacing={self.reference_image.GetSpacing()}")
            print(f"处理后ROI: Spacing={roi_image.GetSpacing()}")
            
            # 2. 对每个特征进行处理
            results = {}
            for feature_name in feature_names:
                try:
                    print(f"\n处理特征: {feature_name}")
                    results[feature_name] = {}
                    
                    # 计算特征图
                    feature_map = self._generate_feature_map(
                        image_path, 
                        sitk.GetImageFromArray(roi_array), 
                        feature_name,
                        kernel_radius
                    )
                    
                    if feature_map is None:
                        results[feature_name] = {pos: None for pos in slice_positions}
                        continue
                    
                    # 对每个位置进行可视化
                    for position in slice_positions:
                        try:
                            # 获取该位置的切片索引
                            slice_idx = self._get_slice_index(roi_array, position)
                            
                            # 可视化该切片的特征图
                            visualization = self._visualize_feature_map(
                                image_path,
                                sitk.GetImageFromArray(roi_array),
                                feature_map,
                                slice_idx,
                                feature_name,
                                save_to_file,
                                output_dir
                            )
                            
                            results[feature_name][position] = visualization
                            print(f"位置 {position:.2f} (切片 {slice_idx}) 处理完成")
                            
                        except Exception as e:
                            print(f"处理位置 {position:.2f} 时出错: {str(e)}")
                            results[feature_name][position] = None
                            
                except Exception as e:
                    print(f"处理特征 {feature_name} 时出错: {str(e)}")
                    results[feature_name] = {pos: None for pos in slice_positions}
                    
            return results
                
        except Exception as e:
            print(f"处理过程出错: {str(e)}")
            return {feature: {pos: None for pos in slice_positions} for feature in feature_names}
            
    def _preprocess_roi(self, roi_path, reference_image_path):
        """预处理ROI掩码，确保与参考图像空间信息一致"""
        # 读取图像
        roi_image = sitk.ReadImage(roi_path)
        reference_image = sitk.ReadImage(reference_image_path)
        
        # 打印空间信息
        print("预处理前空间信息:")
        print(f"参考图像: Spacing={reference_image.GetSpacing()}, Origin={reference_image.GetOrigin()}")
        print(f"ROI掩码: Spacing={roi_image.GetSpacing()}, Origin={roi_image.GetOrigin()}")
        
        # 检查是否需要重采样
        if roi_image.GetSpacing() != reference_image.GetSpacing():
            print("正在重采样ROI掩码...")
            resample = sitk.ResampleImageFilter()
            resample.SetReferenceImage(reference_image)
            resample.SetInterpolator(sitk.sitkNearestNeighbor)  # 对掩码使用最近邻插值
            resample.SetTransform(sitk.Transform())
            resample.SetDefaultPixelValue(0)
            roi_image = resample.Execute(roi_image)
        
        # 二值化处理
        roi_array = sitk.GetArrayFromImage(roi_image)
        roi_array = (roi_array > 0).astype(np.uint8)
        
        # 创建新的ROI图像并复制参考图像的空间信息
        new_roi = sitk.GetImageFromArray(roi_array)
        new_roi.CopyInformation(reference_image)
        
        print("预处理后空间信息:")
        print(f"ROI掩码: Spacing={new_roi.GetSpacing()}, Origin={new_roi.GetOrigin()}")
        
        return new_roi
        
    def _get_slice_index(self, roi_array, position):
        """获取指定位置的切片索引"""
        if not 0 <= position <= 1:
            raise ValueError("位置参数必须在0到1之间")
            
        roi_areas = np.sum(roi_array, axis=(1, 2))
        roi_slices = np.where(roi_areas > 0)[0]
        
        if len(roi_slices) == 0:
            raise ValueError("掩膜中没有找到ROI区域")
            
        return roi_slices[int(len(roi_slices) * position)]
        
    def _generate_feature_map(self, image_path, roi_image, feature_name, kernel_radius):
        """生成特征图"""
        import logging
        logging.getLogger('radiomics').setLevel(logging.ERROR)
        
        try:
            # 读取原始图像
            original_image = sitk.ReadImage(image_path)
            
            # 确保ROI和图像具有相同的空间信息
            roi_array = sitk.GetArrayFromImage(roi_image)
            aligned_roi = sitk.GetImageFromArray(roi_array)
            aligned_roi.CopyInformation(original_image)
            
            # 设置基本特征提取参数 (IBSI标准)
            setting_dict = {
                # 基本参数
                'label': 1,                        # ROI标签值
                'interpolator': 'sitkBSpline',     # IBSI推荐的B样条插值
                'correctMask': True,
                'geometryTolerance': 1e-6,         # IBSI建议的几何容差
                
                # 图像预处理参数 (IBSI标准)
                'normalize': True,                 
                'normalizeScale': 100,             # 标准化到100的范围
                'removeOutliers': None,            # IBSI不建议移除异常值
                'binWidth': 25,                    # IBSI建议的标准bin宽度
                'resampledPixelSpacing': original_image.GetSpacing(),  # 明确指定使用原始spacing
                'preCrop': False,                  # IBSI建议保持原始图像大小
                'force2D': True,
                'force2Ddimension': 0,
                'correctMask': True,
                
                # GLCM特定参数 (IBSI标准)
                'force2D': True,                   
                'force2Ddimension': 0,
                'distances': [1],                  # IBSI标准距离
                'symmetrical': True,               # IBSI要求对称化
                'weightingNorm': None,
                
                # 禁用不必要的重采样
                'interpolator': 'sitkBSpline',
                'resampledPixelSpacing': None,
                
                # 计算参数
                'voxelBatch': 1000,               # 提高计算效率
                'maskedKernel': True,
                'kernelRadius': kernel_radius,     # 根据具体应用调整
                'verbose': True
            }
            
            # 创建特征提取器
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
                    'angles': [0, np.pi/4, np.pi/2, 3*np.pi/4]
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
                    # 1. 图像预处理参数
                    'normalize': True,                 
                    'normalizeScale': 100,             
                    'removeOutliers': None,            # 暂时禁用异常值移除
                    
                    # 2. GLCM特定参数调整
                    'force2D': True,
                    'force2Ddimension': 0,
                    'distances': [1],                  # 只使用一个距离
                    'angles': [0],                     # 只使用一个角度
                    
                    # 3. 特征计算参数
                    'binWidth': 25,                    # 减小bin宽���，增加灰度级别
                    'voxelBatch': 1000,
                    'kernelRadius': 5,  # 限制核半径
                    
                    # 4. 其他优化参数
                    'preCrop': True,
                    'label': 1,
                    'geometryTolerance': 0.0001,      # 提高几何精度
                })
            
            # 启用相应的特征类别和具体特征
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
                
            elif 'glszm' in feature_name:
                extractor.enableFeatureClassByName('glszm')
                if 'GrayLevelVariance' in feature_name:
                    extractor.enableFeaturesByName(glszm=['GrayLevelVariance'])

            elif 'shape' in feature_name:
                extractor.enableFeatureClassByName('shape')
                if 'SurfaceArea' in feature_name:
                    extractor.enableFeaturesByName(shape=['SurfaceArea'])
                # 可以根据需要添加更多形状特征的启用逻辑
            
            # 打印调试信息
            print(f"特征提取器配置:")
            print(f"- 图像类型: {extractor.enabledImagetypes}")
            print(f"- 启用的特征: {extractor.enabledFeatures}")
            
            # 执行特征提取
            print(f"开始提取特征: {feature_name}")
            
            result = extractor.execute(original_image, aligned_roi, voxelBased=True)
            
            # 检查结果中的特征图
            for key in result.keys():
                if isinstance(result[key], sitk.Image):
                    feature_key = key
                    break
                    
            print(f"找到特征图: {feature_key}")
            feature_map = result[feature_key]
            
            # 处理特征图
            for key, val in result.items():
                if isinstance(val, sitk.Image):
                    print(f"处理特征图: {key}")
                    try:
                        # 获取特征图
                        feature_map = val
                        
                        # 打印尺寸信息
                        print(f"原始图像尺寸: {original_image.GetSize()}")
                        print(f"特征图尺寸: {feature_map.GetSize()}")
                        print(f"ROI掩码尺寸: {roi_image.GetSize()}")
                        
                        # 重采样特征图到原始图像尺寸
                        resample = sitk.ResampleImageFilter()
                        resample.SetReferenceImage(original_image)
                        resample.SetInterpolator(sitk.sitkLinear)
                        resample.SetTransform(sitk.Transform())
                        resample.SetDefaultPixelValue(0)
                        
                        feature_map_resampled = resample.Execute(feature_map)
                        
                        print(f"重采样后特征图尺寸: {feature_map_resampled.GetSize()}")
                        
                        # 确保特征图具有正确的空间信息
                        feature_array = sitk.GetArrayFromImage(feature_map_resampled)
                        feature_map_aligned = sitk.GetImageFromArray(feature_array)
                        feature_map_aligned.CopyInformation(original_image)
                        
                        # 应用ROI掩码
                        feature_array = sitk.GetArrayFromImage(feature_map_aligned)
                        roi_array = sitk.GetArrayFromImage(aligned_roi)
                        
                        # 验证数组形状
                        print(f"特征数组形状: {feature_array.shape}")
                        print(f"ROI数组形状: {roi_array.shape}")
                        
                        if feature_array.shape != roi_array.shape:
                            raise ValueError(f"数组形状不匹配: 特征图 {feature_array.shape} vs ROI {roi_array.shape}")
                        
                        # 只处理ROI区域内的值
                        valid_mask = roi_array > 0
                        if not valid_mask.any():
                            print("警告: ROI掩码中没有有效区域")
                            return None
                        
                        # 打印值范围信息
                        roi_values = feature_array[valid_mask]
                        if len(roi_values) > 0:
                            print(f"特征值统计:")
                            print(f"- 原始非零值数量: {np.sum(feature_array != 0)}")
                            print(f"- 原始值范围: [{np.min(feature_array)}, {np.max(feature_array)}]")
                            print(f"- ROI内非零值数量: {np.sum(roi_values != 0)}")
                            print(f"- ROI内值范围: [{np.min(roi_values)}, {np.max(roi_values)}]")
                        
                        # 归一化处理
                        if valid_mask.any():
                            valid_values = feature_array[valid_mask]
                            if len(valid_values) > 0 and not np.all(valid_values == 0):
                                p2, p98 = np.percentile(valid_values[valid_values != 0], (2, 98))
                                if p98 > p2:
                                    feature_array[valid_mask] = np.clip(feature_array[valid_mask], p2, p98)
                                    feature_array[valid_mask] = (feature_array[valid_mask] - p2) / (p98 - p2)
                        
                        # 应用掩码
                        feature_array[~valid_mask] = 0
                        
                        # 返回对齐后的特征图
                        result_image = sitk.GetImageFromArray(feature_array)
                        result_image.CopyInformation(original_image)
                        return result_image
                        
                    except Exception as e:
                        print(f"特征图处理出错: {str(e)}")
                        return None
            
            print("未找到可视化特征图")
            return None
            
        except Exception as e:
            print(f"特征提取出错: {str(e)}")
            return None
        
    def _visualize_feature_map(self, image_path, roi_image, feature_map, slice_idx, feature_name, save_to_file=False, output_dir=None):
        """可视化特征图并返回base64编码的PNG图像"""
        try:
            # 读取原始图像和特征图数据
            original_image = sitk.ReadImage(image_path)
            background_array = sitk.GetArrayFromImage(original_image)[slice_idx]
            feature_array = sitk.GetArrayFromImage(feature_map)[slice_idx]
            roi_array = sitk.GetArrayFromImage(roi_image)[slice_idx]
   
            # 打印调试信息
            print(f"\n可视化处理信息:")
            print(f"- 背景图形状: {background_array.shape}")
            print(f"- 特征图形状: {feature_array.shape}")
            print(f"- ROI形状: {roi_array.shape}")
            print(f"- 特征图spacing: {feature_map.GetSpacing()}")
            print(f"- 原始图像spacing: {original_image.GetSpacing()}")

            # 确保特征图与原始图像具有相同的spacing
            if feature_map.GetSpacing() != original_image.GetSpacing():
                print("正在调整特征图spacing...")
                feature_map_temp = sitk.GetImageFromArray(feature_array)
                feature_map_temp.CopyInformation(original_image)
                feature_array = sitk.GetArrayFromImage(feature_map_temp)

            # 归一化背景图和特征图
            background_array = self._normalize_01(background_array)
            feature_array = self._normalize_01(feature_array)

            # 创建自定义颜色映射
            colors = ['#4B0082', '#0000FF', '#00CED1', '#008000', '#FFD700', '#FFA500', '#FF4500']
            cmap = LinearSegmentedColormap.from_list('custom', colors)

            # 创建RGB数组
            rgb_array = cmap(feature_array)[:, :, :3]  # 去掉alpha通道
            background_rgb = np.stack([background_array] * 3, axis=-1)  # 转换为RGB

            # 在ROI区域外使用背景图像
            rgb_array[roi_array == 0] = background_rgb[roi_array == 0]

            # 创建图像
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            ax.imshow(rgb_array)

            # 设置图像属性
            ax.axis('off')
            ax.set_title(feature_name, fontsize=12, pad=20)

            # 保存到内存缓冲区
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0)

            # 如果需要保存到文件
            if save_to_file and output_dir:
                file_path = os.path.join(output_dir, f"{feature_name}_slice_{slice_idx}.png")
                plt.savefig(file_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
                print(f"特征图已保存到: {file_path}")

            plt.close()

            # 获取base64编码
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            graph = base64.b64encode(image_png).decode('utf-8')
            gc.collect()
            return graph

        except Exception as e:
            print(f"特征图可视化失败: {str(e)}")
            return None
        
    @staticmethod
    def _normalize_01(data, clip=0.0):
        """将数据归一化到0-1范围"""
        new_data = np.asarray(data, dtype=np.float32)
        if np.all(new_data == 0):
            return new_data
        
        if clip > 1e-6:
            data_list = np.sort(data.flatten())
            new_data = np.clip(new_data, 
                              data_list[int(clip * len(data_list))],
                              data_list[int((1 - clip) * len(data_list))])
        
        data_min = np.min(new_data)
        data_max = np.max(new_data)
        if data_max > data_min:
            return (new_data - data_min) / (data_max - data_min)
        return new_data

    def _validate_feature_values(self, feature_array, roi_array, feature_name):
        """验证特征值的有效"""
        valid_mask = roi_array > 0
        if not valid_mask.any():
            print(f"警告: {feature_name} - ROI掩码中没有有效区域")
            return False
        
        roi_values = feature_array[valid_mask]
        if len(roi_values) == 0:
            print(f"警告: {feature_name} - ROI区域内没有有效值")
            return False
        
        if np.all(roi_values == 0):
            print(f"警告: {feature_name} - ROI区域内所有值都为0")
            return False
        
        print(f"特征值统计 ({feature_name}):")
        print(f"- ROI区域值范围: [{np.min(roi_values)}, {np.max(roi_values)}]")
        print(f"- ROI区域非零值数量: {np.sum(roi_values != 0)}")
        print(f"- ROI区域均值: {np.mean(roi_values)}")
        print(f"- ROI区域标准差: {np.std(roi_values)}")
        
        return True

    def _check_inputs(self, original_image, roi_image, feature_name):
        """检查输入数据的有效性"""
        # 检查图像数据
        img_array = sitk.GetArrayFromImage(original_image)
        if np.all(img_array == 0):
            raise ValueError("输入图像全为0")
        
        # 检查ROI数据
        roi_array = sitk.GetArrayFromImage(roi_image)
        if np.sum(roi_array > 0) == 0:
            raise ValueError("ROI掩码中没有有效区域")
        
        # 打印输入数据的基本信息
        print(f"\n输入数据检查 ({feature_name}):")
        print(f"- 图像值范围: [{np.min(img_array)}, {np.max(img_array)}]")
        print(f"- ROI掩码非零像素数: {np.sum(roi_array > 0)}")


def main():
    """测试特征可视化器的功能"""
    try:
        # 初始化可视化器
        visualizer = RadiomicsFeatureVisualizer()
        
        # 设置测试参数
        image_path = r'test_data\negative\image_SJS_0212.nii.gz'
        roi_path = r'test_data\negative\label_SJS_0212.nii.gz'
        save_folder = r'feature_maps\negative'
        base_name = os.path.basename(image_path)[:-7]
        save_path = os.path.join(save_folder, base_name)
        os.makedirs(save_path, exist_ok=True)
        slice_positions = [0.8]  # 切片位置
        feature_names = [
            'original_glrlm_LongRunHighGrayLevelEmphasis',	
            'square_firstorder_Kurtosis', 
            'squareroot_glrlm_RunEntropy', 
            'logarithm_glrlm_LongRunHighGrayLevelEmphasis',
        ]
        # 待选的特征名称
        # feature_names = [
        #     'original_shape_SurfaceArea', 'original_firstorder_Kurtosis',	
        #     'original_glrlm_LongRunHighGrayLevelEmphasis',	'original_glrlm_RunEntropy', 'original_gldm_DependenceNonUniformity', 
        #     'original_gldm_GrayLevelNonUniformity',	'original_glcm_Contrast', 'original_glcm_ClusterShade', 'original_glszm_GrayLevelVariance',	
        #     'square_firstorder_Kurtosis', 'square_glrlm_LongRunHighGrayLevelEmphasis', 'square_glrlm_RunEntropy', 
        #     'square_gldm_DependenceNonUniformity', 'square_gldm_GrayLevelNonUniformity', 'square_glcm_Contrast',	
        #     'square_glcm_ClusterShade',	'square_glszm_GrayLevelVariance',	'squareroot_firstorder_Kurtosis',	'squareroot_glrlm_LongRunHighGrayLevelEmphasis',	
        #     'squareroot_glrlm_RunEntropy', 'squareroot_gldm_DependenceNonUniformity',	'squareroot_gldm_GrayLevelNonUniformity',	
        #     'squareroot_glcm_Contrast', 'squareroot_glcm_ClusterShade', 'squareroot_glszm_GrayLevelVariance', 'logarithm_firstorder_Kurtosis',	
        #     'logarithm_glrlm_LongRunHighGrayLevelEmphasis',	'logarithm_glrlm_RunEntropy',	'logarithm_gldm_DependenceNonUniformity',	
        #     'logarithm_gldm_GrayLevelNonUniformity', 'logarithm_glcm_Contrast', 'logarithm_glcm_ClusterShade', 'logarithm_glszm_GrayLevelVariance'
        # ]

        print("=== 开始特征可视化测试 ===")
        print(f"图像路径: {image_path}")
        print(f"ROI路径: {roi_path}")
        print(f"特征列表: {feature_names}")
        print(f"切片位置: {slice_positions}")
        
        # 生成并可视化特征图
        results = visualizer.generate_and_visualize_features(
            image_path=image_path,
            roi_path=roi_path,
            slice_positions=slice_positions,
            feature_names=feature_names,
            kernel_radius=5,
            save_to_file=True,
            output_dir=save_path
        )
        
        # 显示处理结果
        print("\n=== 处理结果 ===")
        for feature_name, positions in results.items():
            print(f"\n特征: {feature_name}")
            for position, base64_image in positions.items():
                if base64_image:
                    print(f"  位置 {position:.2f} 的特征图生成成功")
                    
                    # 从base64解码并显示图像
                    try:
                        # 解码base64图像
                        image_data = base64.b64decode(base64_image)
                        
                        # 使用BytesIO读取图像数据
                        image_buffer = BytesIO(image_data)
                        
                        # 创建新的图像窗口
                        plt.figure(figsize=(10, 8))
                        plt.imshow(plt.imread(image_buffer))
                        plt.axis('off')
                        plt.title(f"{feature_name} at position {position:.2f}")
                        # plt.show()
                        
                        # 关闭图像和清理内存
                        plt.close()
                        image_buffer.close()
                        gc.collect()
                        
                    except Exception as e:
                        print(f"  显示图像失败: {str(e)}")
                else:
                    print(f"  位置 {position:.2f} 的特征图生成失败")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        # 确保清理所有资源
        plt.close('all')
        gc.collect()

if __name__ == '__main__':
    main() 