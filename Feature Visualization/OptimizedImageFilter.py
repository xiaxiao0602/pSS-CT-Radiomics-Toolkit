import SimpleITK as sitk
import numpy as np
import os
import warnings
from tqdm import tqdm
from radiomics.imageoperations import *
import matplotlib.pyplot as plt
from skimage import measure

class OptimizedImageFilter:
    """
    优化的图像滤波器类
    通过只处理目标切片及必要的相邻切片来提高计算效率
    """
    
    def __init__(self):
        self.image_generators = {
            'Square': (getSquareImage, []),
            'SquareRoot': (getSquareRootImage, []),
            'Logarithm': (getLogarithmImage, []),
        }
    
    @staticmethod
    def preprocess_mask(mask):
        """
        预处理掩膜，将所有非零值统一转换为1
        
        参数：
        mask: SimpleITK.Image, 输入掩膜
        
        返回：
        SimpleITK.Image: 处理后的二值掩膜
        """
        mask_array = sitk.GetArrayFromImage(mask)
        # 将所有非零值转换为1
        binary_mask_array = np.where(mask_array > 0, 1, 0).astype(np.uint8)
        
        # 转回SimpleITK图像并保持原有的图像信息
        binary_mask = sitk.GetImageFromArray(binary_mask_array)
        binary_mask.CopyInformation(mask)
        
        # 打印转换信息
        unique_values = np.unique(mask_array)
        if len(unique_values) > 2:
            print(f"掩膜值已从 {unique_values} 转换为二值 [0, 1]")
            
        return binary_mask
    
    @staticmethod
    def validate_mask(image, mask):
        """
        验证掩膜是否符合基本要求
        
        参数：
        image: SimpleITK.Image, 原始图像
        mask: SimpleITK.Image, ROI掩膜
        """
        # 检查尺寸
        if image.GetSize() != mask.GetSize():
            raise ValueError("掩膜尺寸与原始图像不匹配")
        
        # 检查是否存在ROI
        mask_array = sitk.GetArrayFromImage(mask)
        if np.sum(mask_array) == 0:
            raise ValueError("掩膜中没有ROI区域（全为0）")
        
        # 检查元数据
        for attr in ['Origin', 'Spacing', 'Direction']:
            if getattr(image, f'Get{attr}')() != getattr(mask, f'Get{attr}')():
                raise ValueError(f"{attr}与原始图像不匹配")
    
    @staticmethod
    def get_required_slices(filter_type, sigma=None):
        """
        根据滤波器类型返回需要的切片范围
        
        参数：
        filter_type: str, 滤波器类型
        sigma: float, LoG滤波器的sigma值
        
        返回：
        int: 目标切片上下需要的切片数
        """
        if filter_type == 'LoG':
            padding = int(np.ceil(sigma * 3))
        elif filter_type == 'LBP3D':
            padding = 1
        elif filter_type == 'wavelet':
            padding = 2
        else:
            padding = 0
        return padding
    
    @staticmethod
    def extract_required_volume(image_array, target_slice_idx, padding):
        """
        提取目标切片及其必要的相邻切片
        
        参数：
        image_array: ndarray, 输入图像数组
        target_slice_idx: int, 目标切片索引
        padding: int, 需要的额外切片数
        
        返回：
        tuple: (提取的体积, 目标切片在提取体积中的相对位置)
        """
        start_idx = max(0, target_slice_idx - padding)
        end_idx = min(image_array.shape[0], target_slice_idx + padding + 1)
        return image_array[start_idx:end_idx], target_slice_idx - start_idx
    
    @staticmethod
    def validate_slice_range(image_shape, target_slice_idx, padding):
        """
        验证切片范围是否合理
        
        参数：
        image_shape: tuple, 图像形状
        target_slice_idx: int, 目标切片索引
        padding: int, padding大小
        """
        if target_slice_idx - padding < 0 or target_slice_idx + padding >= image_shape[0]:
            warnings.warn("Padding size might be too large for the given slice index")
    
    @staticmethod
    def get_roi_slice_positions(roi_array, positions=None):
        """
        获取ROI区域在z轴方向上指定位置的切片索引
        
        参数：
        roi_array: ndarray, ROI掩膜数组
        positions: list of float, 指定的位置比例，范围[0-1]，默认为[0.25, 0.5, 0.75]
        
        返回：
        list: 对应位置的切片索引列表
        dict: 位置名称到索引的映射
        """
        # 默认位置如果未指定
        if positions is None:
            positions = [0.25, 0.5, 0.75]
        
        # 验证位置参数
        if not all(0 <= p <= 1 for p in positions):
            raise ValueError("所有位置参数必须在0到1之间")
        
        # 获取每个切片上ROI的面积
        roi_areas = np.sum(roi_array, axis=(1, 2))
        
        # 获取有ROI的切片索引
        roi_slices = np.where(roi_areas > 0)[0]
        
        if len(roi_slices) == 0:
            raise ValueError("掩膜中没有找到ROI区域")
        
        # 计算每个位置的索引
        position_indices = {}
        for pos in positions:
            # 计算索引
            idx = roi_slices[int(len(roi_slices) * pos)]
            # 生成位置名称（例如：将0.25转换为"Pos25"）
            pos_name = f"Pos{int(pos * 100)}"
            position_indices[pos_name] = idx
        
        return position_indices
    
    def generate_filter_images(self, input_image, roi_mask, target_slice_idx=None, use_mask_only=False):
        try:
            # 验证原始掩膜
            self.validate_mask(input_image, roi_mask)
            
            # 预处理掩膜为二值图像
            roi_mask = self.preprocess_mask(roi_mask)
            
            # 转换为数组
            img_array = sitk.GetArrayFromImage(input_image)
            roi_array = sitk.GetArrayFromImage(roi_mask)
            
            images = {}
            
            # 定义软组织窗参数
            # WW = 400  # 窗宽
            # WC = 40   # 窗位
            WW = 2200  # 窗宽
            WC = 1500   # 窗位
            window_min = WC - WW/2  # -160 HU/400
            window_max = WC + WW/2  # 240 HU/2600
            
            for name, (generator, params) in self.image_generators.items():
                try:
                    # 获取单个切片
                    slice_img_array = img_array[target_slice_idx:target_slice_idx+1]
                    slice_roi_array = roi_array[target_slice_idx:target_slice_idx+1]
                    
                    # 应用软组织窗归一化
                    # 步骤1：将值限制在窗宽范围内
                    slice_img_array = np.clip(slice_img_array, window_min, window_max)
                    # 步骤2：线性归一化到[0,1]
                    slice_img_array = (slice_img_array - window_min) / (window_max - window_min)
                    
                    # 步骤3：将ROI外的区域置零
                    if use_mask_only:
                        slice_img_array = slice_img_array * slice_roi_array
                    
                    # 创建新的图像对象（使用float类型）
                    slice_img = sitk.GetImageFromArray(slice_img_array.astype(np.float32))
                    slice_roi = sitk.GetImageFromArray(slice_roi_array)
                    
                    # 设置正确的spacing和direction
                    slice_img.SetSpacing(input_image.GetSpacing())
                    slice_img.SetDirection(input_image.GetDirection())
                    slice_img.SetOrigin(input_image.GetOrigin())
                    
                    slice_roi.SetSpacing(input_image.GetSpacing())
                    slice_roi.SetDirection(input_image.GetDirection())
                    slice_roi.SetOrigin(input_image.GetOrigin())
                    
                    # 生成滤波图像
                    result = next(generator(slice_img, slice_roi, 
                                        maskOnly=False,  # 不再需要maskOnly参数
                                        verbose=False))[0]
                    
                    # 确保结果图像具有正确的空间信息
                    result_array = sitk.GetArrayFromImage(result)
                    
                    # 再次应用ROI掩膜确保边界正确
                    if use_mask_only:
                        result_array = result_array * slice_roi_array
                    
                    filtered_image = sitk.GetImageFromArray(result_array)
                    
                    # 设置正确的空间信息
                    filtered_image.SetSpacing(input_image.GetSpacing())
                    filtered_image.SetDirection(input_image.GetDirection())
                    filtered_image.SetOrigin(input_image.GetOrigin())
                    
                    images[name] = filtered_image
                    
                except Exception as e:
                    print(f"处理 {name} 滤波器时出错: {str(e)}")
                    continue
            
            return images
            
        except Exception as e:
            print(f"处理过程出错: {str(e)}")
            return {}
    
    def generate_filter_images_for_positions(self, input_image, roi_mask, positions=None, use_mask_only=False):
        """
        在ROI的指定位置生成滤波图像并返回
        """
        try:
            # 验证原始掩膜
            self.validate_mask(input_image, roi_mask)
            
            # 预处理掩膜为二值图像
            roi_mask = self.preprocess_mask(roi_mask)
            
            # 获取数组
            roi_array = sitk.GetArrayFromImage(roi_mask)
            
            # 获取特定位置的切片索引
            position_indices = self.get_roi_slice_positions(roi_array, positions)
            
            all_filtered_images = {}
            
            for position_name, idx in position_indices.items():
                print(f"\n处理 {position_name} 位置 (切片索引: {idx})")
                
                # 传递use_mask_only参数
                filtered_images = self.generate_filter_images(
                    input_image, roi_mask, 
                    target_slice_idx=idx,
                    use_mask_only=use_mask_only)
                
                # 将结果添加到总字典中
                all_filtered_images[position_name] = filtered_images
            
            return all_filtered_images, position_indices
            
        except Exception as e:
            warnings.warn(f"处理过程出错: {str(e)}")
            return {}, {}

def main():
    """使用示例"""
    # 设置输入路径
    num = 126
    # input_path = rf'test_data\negative\image_SJS_0{num}.nii.gz'
    # roi_path = rf'test_data\negative\label_SJS_0{num}.nii.gz'
    input_path = rf'test_data\positive\image_SJS_0{num}.nii.gz'
    roi_path = rf'test_data\positive\label_SJS_0{num}.nii.gz'
    base_name = os.path.basename(input_path)
    save_folder = os.path.join(r'feature_maps', base_name)
    os.makedirs(save_folder, exist_ok=True)
    
    # 设置参数
    custom_positions = [0.2, 0.4, 0.6, 0.8]  # 可选：指定位置
    use_mask_only = False  # 可选：是否只处理ROI区域
    
    # 读取图像
    input_image = sitk.ReadImage(input_path)
    roi_mask = sitk.ReadImage(roi_path)
    
    # 打印图像和掩膜的空间信息
    print("输入图像的Spacing:", input_image.GetSpacing())
    print("输入图像的Origin:", input_image.GetOrigin())
    print("输入图像的Direction:", input_image.GetDirection())
    
    print("ROI掩膜的Spacing:", roi_mask.GetSpacing())
    print("ROI掩膜的Origin:", roi_mask.GetOrigin())
    print("ROI掩膜的Direction:", roi_mask.GetDirection())
    
    # 创建滤波器实例
    filter_generator = OptimizedImageFilter()
    
    # 生成滤波图像
    filtered_images, position_indices = filter_generator.generate_filter_images_for_positions(
        input_image, 
        roi_mask, 
        positions=custom_positions,
        use_mask_only=use_mask_only)
    
    # 打印结果统计
    for position, images in filtered_images.items():
        print(f"\n{position} 位置生成了 {len(images)} 个滤波图像")
        if use_mask_only:
            print("（仅处理ROI区域）")
        else:
            print("（处理全局图像）")

    # 可视化滤波图像
    for position, images in filtered_images.items():
        if not images:  # 如果该位置没有生成任何图像，跳过
            continue
            
        # 获取该位置的原始图像切片和ROI掩膜
        slice_idx = position_indices[position]
        original_slice = sitk.GetArrayFromImage(input_image)[slice_idx]
        roi_slice = sitk.GetArrayFromImage(roi_mask)[slice_idx]
        
        # 保存原始图像（应用CT窗）
        if save_folder:
            save_dir = os.path.join(save_folder, position)
            os.makedirs(save_dir, exist_ok=True)
            
            # 找到ROI边界
            contours = measure.find_contours(roi_slice, 0.5)
            
            # 原始图像使用软组织窗
            # ww = 400  # 窗宽
            # wc = 40   # 窗位
            ww = 2200  # 窗宽
            wc = 1500   # 窗位
            ct_vmin = wc - ww/2
            ct_vmax = wc + ww/2

            # 保存原始图像（应用CT窗）
            plt.figure(figsize=(5, 4), dpi=300)
            masked_original = original_slice
            
            # 绘制原始图像
            plt.imshow(masked_original, cmap='gray', vmin=ct_vmin, vmax=ct_vmax)
            
            # 绘制ROI边界
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
            
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, 'original.png'), 
                       bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

        # 处理滤波图像（自动调整显示范围）
        for filter_name, img in images.items():
            img_array = sitk.GetArrayFromImage(img)[0]
            # masked_img_array = img_array * roi_slice
            masked_img_array = img_array

            # 计算滤波图像的显示范围（排除零值）
            non_zero_values = masked_img_array[masked_img_array != 0]
            if non_zero_values.size > 0:
                p1 = np.percentile(non_zero_values, 1)
                p99 = np.percentile(non_zero_values, 99)
                vmin = max(0, p1)  # 确保最小值为0或正数
                vmax = p99
            else:
                vmin, vmax = 0, 1  # 默认范围

            # 保存滤波图像
            plt.figure(figsize=(5, 4), dpi=300)
            plt.imshow(masked_img_array, 
                      cmap='gray', 
                      vmin=vmin, 
                      vmax=vmax)
            
            # 绘制ROI边界（使用相同的轮廓数据）
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='red')
            
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f'{filter_name}.png'),
                      bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
        
        # 计算需要的行数和列数
        n_images = len(images)
        n_cols = min(3, n_images)  # 每行最多3张图
        n_rows = (n_images + n_cols - 1) // n_cols
        
        # 创建图像网格
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), dpi=150)
        fig.suptitle(f'滤波图像 - {position}位置', fontsize=16)
        
        # 确保axes是二维的
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 隐藏空白子图
        for idx in range(n_images, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        # 调整子图之间的间距
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存组合图像（中等质量）
        if save_folder:
            plt.savefig(os.path.join(save_dir, 'filtered_images.png'), dpi=150)
        
        plt.close()

if __name__ == '__main__':
    main()
