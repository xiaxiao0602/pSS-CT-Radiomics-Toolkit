import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

def create_vertical_colorbar(save_path=r"3D_feature_maps/colorbar.png"):
    """
    创建并保存垂直颜色条
    
    参数：
    save_path: 保存路径，默认当前目录下的colorbar.png
    """
    # 使用与3D可视化相同的颜色配置
    colors = ['#4B0082', '#0000FF', '#00CED1', '#008000', 
             '#FFD700', '#FFA500', '#FF4500']
    
    # 创建颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(2, 6))  # 竖长条形尺寸
    fig.subplots_adjust(left=0.3, right=0.5)  # 调整边距
    
    # 生成颜色条数据
    gradient = np.linspace(1, 0, 256).reshape(256, 1)  # 从1到0生成渐变
    
    # 显示颜色条
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    
    # 隐藏坐标轴
    ax.set_axis_off()
    
    # 保存图像
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    print(f"颜色条已保存至 {save_path}")
    plt.close()

if __name__ == "__main__":
    create_vertical_colorbar() 