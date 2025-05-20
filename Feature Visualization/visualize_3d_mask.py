import vedo
import SimpleITK as sitk
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os

def visualize_3d_feature_map(nii_path, roi_mask_path, opacity=0.5, threshold=0.1, save_path=None, camera_pos=None, view_up=None):
    """
    三维可视化特征图
    
    Args:
        nii_path: nii文件路径
        roi_mask_path: ROI掩码文件路径，非0区域为ROI
        opacity: 透明度 (0-1)
        threshold: 显示阈值，小于此值的体素将被忽略
        save_path: 保存图像的路径
        camera_pos: 相机位置
        view_up: 视角向上方向
    """
    # 读取nii图像和ROI掩码
    img = sitk.ReadImage(nii_path)
    img_array = sitk.GetArrayFromImage(img)
    
    roi_mask = sitk.ReadImage(roi_mask_path)
    roi_mask_array = sitk.GetArrayFromImage(roi_mask).astype(bool)
    
    # 获取spacing信息 (x,y,z) -> (z,y,x)
    spacing = img.GetSpacing()[::-1]
    
    print("Spacing:", spacing)
    print("图像形状:", img_array.shape)
    
    # # 归一化数据到0-1范围
    # if np.max(img_array) > np.min(img_array):
    #     img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    
    # 打印数据信息
    print("值范围:", np.min(img_array), np.max(img_array))
    print("非零值数量:", np.sum(img_array > 0))
    
    # 创建自定义颜色映射
    colors = ['#4B0082', '#0000FF', '#00CED1', '#008000', '#FFD700', '#FFA500', '#FF4500']
    
    # 预处理数据：在ROI区域内替换0值为最小非零值
    if np.any(img_array > 0):
        min_nonzero = np.min(img_array[img_array > 0])
        # 创建处理后的数组
        processed_array = np.where(
            roi_mask_array & (img_array == 0),  # ROI区域内的0值
            min_nonzero,  # 替换为最小非零值
            img_array     # 其他区域保持原值
        )
    else:
        processed_array = img_array.copy()
    
    # 修改这里：将vedo的Plotter重命名为vedo_plt
    vedo_plt = vedo.Plotter(bg='white')
    
    # 创建体积，考虑spacing
    vol = vedo.Volume(processed_array, spacing=spacing)
    
    # 设置颜色和透明度映射
    data_min = np.min(processed_array[processed_array > 0]) if np.any(processed_array > 0) else 0
    data_max = np.max(processed_array)
    
    # 调整透明度映射：
    # - ROI区域内的0值（已替换为data_min）保持不透明
    # - ROI区域外的0值保持完全透明
    vol.alpha([(0, 0), (data_min, opacity), (data_max, opacity)])  # 三段式透明度映射
    vol.alpha_unit(opacity)
    vol.mode(1)
    vol.cmap(colors)
    
    # 可选：设置显示范围（如果需要限制显示范围）
    # vol.mapper().SetScalarRange(min_val, max_val)  # 根据实际数据范围设置
    
    # 添加到场景
    vedo_plt.add(vol)
    
    # 计算实际的物理尺寸
    physical_size = np.array(img_array.shape) * np.array(spacing)
    max_dim = max(physical_size)
    
    # 设置相机位置（根据物理尺寸调整）
    if camera_pos is not None:
        # 根据物理尺寸缩放相机位置
        camera_pos = np.array(camera_pos) * max_dim
        vedo_plt.camera.SetPosition(*camera_pos)
        if view_up is not None:
            vedo_plt.camera.SetViewUp(*view_up)
    else:
        vedo_plt.camera.SetPosition(max_dim, max_dim, max_dim)
        vedo_plt.camera.SetViewUp(0, 0, 1)
    
    # 添加坐标轴（使用物理尺寸，注意轴的顺序）
    ax = vedo.Axes(
        xrange=(0, physical_size[0]),  # Z轴
        yrange=(0, physical_size[1]),  # Y轴
        zrange=(0, physical_size[2]),  # X轴
        xtitle='Z (mm)', 
        ytitle='Y (mm)', 
        ztitle='X (mm)',
        xygrid=True,
        yzgrid=True,
        zxgrid=True,
        axes_linewidth=2,    # 坐标轴线宽度
        grid_linewidth=0.1,  # 网格线宽度
        xygrid_color='black',  # XY平面网格颜色
        yzgrid_color='black',  # YZ平面网格颜色
        zxgrid_color='black',  # ZX平面网格颜色
        xyalpha=0.1,  # XY平面透明度
        yzalpha=0.1,  # YZ平面透明度
        zxalpha=0.1,  # ZX平面透明度
        c='black'     # 坐标轴颜色
    )
    vedo_plt.add(ax)
    
    # 新增函数：获取ROI切片位置
    def get_roi_slice_positions(roi_array, positions=[0.2, 0.4, 0.6, 0.8]):
        roi_areas = np.sum(roi_array, axis=(1, 2))
        roi_slices = np.where(roi_areas > 0)[0]
        return {f"Pos{int(p*100)}": roi_slices[int(len(roi_slices)*p)] for p in positions}

    # 修改：保存多平面切片
    if save_path:
        import matplotlib.pyplot as plt
        
        # 创建切片保存目录
        slice_dir = os.path.join(os.path.dirname(save_path), "slices")
        os.makedirs(slice_dir, exist_ok=True)
        
        # 获取ROI位置
        position_indices = get_roi_slice_positions(roi_mask_array)
        
        # 创建颜色映射（与3D可视化一致）
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        # 保存每个位置的切片
        for pos_name, slice_idx in position_indices.items():
            plt.figure(figsize=(5, 4), dpi=300)
            
            # 获取当前切片数据和对应的ROI掩码
            current_slice = img_array[slice_idx]
            current_roi = roi_mask_array[slice_idx]
            
            # 创建带透明度的显示
            normalized_slice = np.zeros_like(current_slice, dtype=float)
            roi_mask = current_roi > 0
            
            # 获取整个体积的非零最小值
            min_nonzero_value = np.min(img_array[img_array > 0]) if np.any(img_array > 0) else 0
            max_value = np.max(img_array) if np.max(img_array) > 0 else 1  # 避免除以0
            
            # ROI区域内的处理
            normalized_slice[roi_mask] = np.where(
                current_slice[roi_mask] > 0,
                current_slice[roi_mask] / max_value,  # 非零值正常归一化
                min_nonzero_value / max_value  # 零值显示为全局最小非零值
            )
            
            # ROI区域外的处理
            normalized_slice[~roi_mask] = np.where(
                current_slice[~roi_mask] > 0,
                current_slice[~roi_mask] / max_value,
                np.nan  # ROI外的零值保持透明
            )
            
            # 设置透明度：ROI区域内完全不透明，ROI区域外非零值半透明
            alpha_mask = np.where(roi_mask, 1.0,  # ROI区域内完全不透明
                                np.where(current_slice > 0, opacity, 0))  # ROI区域外非零值半透明
            
            # 绘制图像
            plt.imshow(normalized_slice,
                      cmap=cmap,
                      vmin=0,
                      vmax=1,
                      alpha=alpha_mask)
            
            plt.axis('off')
            slice_path = os.path.join(slice_dir, f"{os.path.basename(nii_path)}_{pos_name}.png")
            plt.savefig(slice_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            print(f"已保存 {pos_name} 切片到 {slice_path}")

    # 如果需要保存图像
    if save_path:
        # 先显示场景
        vedo_plt.show(interactive=False)  # 不进入交互模式
        # 保存当前视角的图像
        vedo_plt.screenshot(save_path,scale=2)
        # 关闭场景
        vedo_plt.close()
    else:
        # 正常显示场景（交互模式）
        vedo_plt.show(interactive=True).close()

def main():
    # 使用示例
    # nii_path = r"3D_feature_maps\original_glrlm_LongRunHighGrayLevelEmphasis.nii.gz"
    # nii_path = r"3D_feature_maps\square_firstorder_Kurtosis.nii.gz"
    # nii_path = r"3D_feature_maps\squareroot_glrlm_RunEntropy.nii.gz"
    # nii_path = r"3D_feature_maps\logarithm_glrlm_LongRunHighGrayLevelEmphasis.nii.gz"
    nii_dir = r"3D_feature_maps\image_SJS_0134.nii.gz"

    # 定义不同视角和保存路径
    views = {
        # 'front': {'pos': (0, -1, 0), 'up': (1, 0, 0)},    # 正视图
        # 'side': {'pos': (0, 0, 1), 'up': (1, 0, 0)},      # 侧视图
        # 'top': {'pos': (1, 0, 0), 'up': (0, 1, 0)},       # 俯视图
        'iso': {'pos': (1, 1, 1), 'up': (1, 0, 0)}        # 等轴测图
    }
    
    # 创建保存目录
    save_folder = os.path.join(nii_dir, r"3D_feature_maps_views")
    os.makedirs(save_folder, exist_ok=True)
    
    for file in os.listdir(nii_dir):
        if file.endswith('nii.gz'):
            nii_path = os.path.join(nii_dir, file)
            # 保存不同视角的图像
            for view_name, view_params in views.items():
                save_path = os.path.join(save_folder, f'{os.path.basename(nii_path)}_{view_name}.png')
                visualize_3d_feature_map(
                    nii_path, 
                    r"test_data\positive\label_SJS_0134.nii.gz",
                    opacity=0.5, 
                    threshold=0, 
                    save_path=save_path,
                    camera_pos=view_params['pos'],
                    view_up=view_params['up']
                )
                print(f"已保存 {view_name} 视角的图像到 {save_path}")

    # # 显示交互式视图
    # visualize_3d_feature_map(nii_path, opacity=1.0, threshold=0)

if __name__ == "__main__":
    main() 