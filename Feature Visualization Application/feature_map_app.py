import sys
import os
import numpy as np
import SimpleITK as sitk
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QComboBox, QLabel, QFileDialog,
                            QSplitter, QFrame, QSlider)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap
from vedo import Plotter, Volume, Axes
import vedo
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class FeatureMapVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D特征图可视化器")
        self.setMinimumSize(1200, 800)
        
        # 初始化变量
        self.feature_files = []
        self.mask_file = None
        self.folder_path = None
        self.current_feature = None
        
        # 初始化缓存字典
        self.negative_cache = {}  # 格式: {(feature_name, opacity): (processed_array, spacing)}
        self.positive_cache = {}  # 格式: {(feature_name, opacity): (processed_array, spacing)}
        
        # 初始化UI
        self.init_ui()
    
    def init_ui(self):
        # 创建主窗口布局
        main_layout = QHBoxLayout()
        
        # 创建左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setMinimumWidth(400)  # 增加最小宽度
        control_panel.setMaximumWidth(900)  # 增加最大宽度
        
        # 添加选择文件夹按钮
        self.folder_btn = QPushButton("选择特征图文件夹")
        self.folder_btn.clicked.connect(self.select_folder)
        control_layout.addWidget(self.folder_btn)
        
        # 添加文件夹路径标签
        self.folder_label = QLabel("未选择文件夹")
        self.folder_label.setWordWrap(True)
        control_layout.addWidget(self.folder_label)
        
        # 添加ROI蒙版文件标签
        self.mask_label = QLabel("未找到ROI蒙版文件")
        self.mask_label.setWordWrap(True)
        control_layout.addWidget(self.mask_label)
        
        # 添加特征选择下拉框
        self.feature_label = QLabel("选择特征:")
        control_layout.addWidget(self.feature_label)
        
        self.feature_combo = QComboBox()
        self.feature_combo.setMinimumWidth(380)  # 设置下拉框最小宽度
        self.feature_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)  # 自动调整宽度以适应内容
        self.feature_combo.currentIndexChanged.connect(self.feature_changed)
        control_layout.addWidget(self.feature_combo)
        
        # 添加透明度下拉框
        self.opacity_label = QLabel("选择透明度:")
        control_layout.addWidget(self.opacity_label)
        
        self.opacity_combo = QComboBox()
        self.opacity_combo.setMinimumWidth(100)
        opacity_values = [str(round(i * 0.1, 1)) for i in range(1, 10)]  # 0.1到1.0，步长0.1
        self.opacity_combo.addItems(opacity_values)
        self.opacity_combo.setCurrentText("0.5")  # 默认值0.5
        self.opacity_combo.currentTextChanged.connect(self.opacity_changed)
        control_layout.addWidget(self.opacity_combo)
        
        # 添加特征预测方向说明框
        direction_label = QLabel("特征预测方向说明:")
        direction_label.setStyleSheet("margin-top: 5px;")
        control_layout.addWidget(direction_label)
        
        # 创建说明文本
        directions_text = """高值指向阴性的特征:
• square_glcm_MaximumProbability

高值指向阳性的特征:
• square_glcm_DifferenceAverage
• square_glrlm_RunPercentage
• square_glrlm_ShortRunEmphasis"""
        directions_info = QLabel(directions_text)
        directions_info.setWordWrap(True)  # 允许文本换行
        directions_info.setStyleSheet("background-color: #f0eba8; padding: 10px; border-radius: 5px;")
        control_layout.addWidget(directions_info)
        
        # 添加参考示例标签
        reference_label = QLabel("参考示例:")
        reference_label.setStyleSheet("margin-top: 15px; font-weight: bold;")
        control_layout.addWidget(reference_label)
        
        # 创建参考示例容器
        reference_container = QWidget()
        reference_layout = QHBoxLayout()
        reference_container.setLayout(reference_layout)
        
        # 创建阴性示例窗口
        negative_container = QWidget()
        negative_layout = QVBoxLayout()
        negative_container.setLayout(negative_layout)
        
        negative_label = QLabel("阴性示例")
        negative_label.setAlignment(Qt.AlignCenter)
        negative_layout.addWidget(negative_label)
        
        self.negative_frame = QFrame()
        self.negative_frame.setMinimumSize(300, 300)  # 设置最小尺寸为正方形
        self.negative_frame.setStyleSheet("border: 1px solid #cccccc;")
        negative_vtk_layout = QVBoxLayout()
        self.negative_frame.setLayout(negative_vtk_layout)
        
        self.negative_widget = QVTKRenderWindowInteractor()
        negative_vtk_layout.addWidget(self.negative_widget)
        negative_layout.addWidget(self.negative_frame)
        
        # 创建阳性示例窗口
        positive_container = QWidget()
        positive_layout = QVBoxLayout()
        positive_container.setLayout(positive_layout)
        
        positive_label = QLabel("阳性示例")
        positive_label.setAlignment(Qt.AlignCenter)
        positive_layout.addWidget(positive_label)
        
        self.positive_frame = QFrame()
        self.positive_frame.setMinimumSize(300, 300)  # 设置最小尺寸为正方形
        self.positive_frame.setStyleSheet("border: 1px solid #cccccc;")
        positive_vtk_layout = QVBoxLayout()
        self.positive_frame.setLayout(positive_vtk_layout)
        
        self.positive_widget = QVTKRenderWindowInteractor()
        positive_vtk_layout.addWidget(self.positive_widget)
        positive_layout.addWidget(self.positive_frame)
        
        # 将阴性和阳性示例添加到参考容器
        reference_layout.addWidget(negative_container)
        reference_layout.addWidget(positive_container)
        
        # 添加参考示例到控制面板
        control_layout.addWidget(reference_container)
        
        # 初始化参考示例的渲染器
        self.negative_plt = Plotter(qt_widget=self.negative_widget, offscreen=False, interactive=True)
        self.positive_plt = Plotter(qt_widget=self.positive_widget, offscreen=False, interactive=True)
        
        # 设置参考示例的背景颜色
        self.negative_plt.background('white')
        self.positive_plt.background('white')
        
        # 初始化参考示例的交互器
        self.negative_iren = self.negative_widget.GetRenderWindow().GetInteractor()
        self.positive_iren = self.positive_widget.GetRenderWindow().GetInteractor()
        self.negative_iren.Initialize()
        self.positive_iren.Initialize()
        
        # 添加空白区域
        control_layout.addStretch()
        
        # 创建中间的VTK可视化窗口
        self.vtk_frame = QFrame()
        vtk_layout = QVBoxLayout()
        self.vtk_frame.setLayout(vtk_layout)
        self.vtk_frame.setMinimumSize(600, 600)  # 设置最小尺寸为正方形
        
        self.vtk_widget = QVTKRenderWindowInteractor()
        vtk_layout.addWidget(self.vtk_widget)
        
        # 创建右侧颜色条面板
        colorbar_panel = QWidget()
        colorbar_layout = QVBoxLayout()
        colorbar_panel.setLayout(colorbar_layout)
        colorbar_panel.setFixedWidth(60)  # 设置固定宽度
        
        # 添加顶部空白区域实现竖直居中
        colorbar_layout.addStretch(1)
        
        # 添加颜色条图例
        colorbar_label = QLabel()
        colorbar_pixmap = QPixmap('colorbar.png')
        # 计算合适的高度以保持1:14的比例
        scaled_height = int(colorbar_panel.width() * 14)
        # 缩放图像保持比例
        colorbar_pixmap = colorbar_pixmap.scaled(
            colorbar_panel.width() - 10,  # 留出一些边距
            scaled_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        colorbar_label.setPixmap(colorbar_pixmap)
        colorbar_label.setAlignment(Qt.AlignCenter)
        colorbar_layout.addWidget(colorbar_label)
        
        # 添加底部空白区域实现竖直居中
        colorbar_layout.addStretch(1)
        
        # 将所有面板添加到主布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.vtk_frame, 1)  # 1表示拉伸因子
        main_layout.addWidget(colorbar_panel)
        
        # 创建中央窗口小部件并设置布局
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 初始化交互器
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.iren.Initialize()
        
        # 初始化主窗口的渲染器
        self.vedo_plt = Plotter(qt_widget=self.vtk_widget, offscreen=False, interactive=True)
        
        # 设置背景颜色为白色
        self.vedo_plt.background('white')
    
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择特征图文件夹")
        if folder_path:
            self.folder_path = folder_path
            self.folder_label.setText(f"已选择文件夹: {folder_path}")
            self.load_files()
    
    def load_files(self):
        """加载文件夹中的所有nii文件"""
        if not self.folder_path:
            return
        
        # 清空当前列表
        self.feature_files = []
        self.feature_combo.clear()
        
        # 加载所有.nii和.nii.gz文件
        files = os.listdir(self.folder_path)
        
        # 首先查找mask文件
        self.mask_file = None
        for f in files:
            if f.lower() == 'mask.nii.gz' or f.lower() == 'mask.nii':
                self.mask_file = os.path.join(self.folder_path, f)
                self.mask_label.setText(f"已找到ROI蒙版: {f}")
                break
        
        # 如果没有找到mask文件，尝试寻找包含'mask'的文件
        if not self.mask_file:
            for f in files:
                if ('mask' in f.lower() or 'roi' in f.lower()) and (f.endswith('.nii.gz') or f.endswith('.nii')):
                    self.mask_file = os.path.join(self.folder_path, f)
                    self.mask_label.setText(f"已找到ROI蒙版: {f}")
                    break
        
        # 如果仍未找到mask文件
        if not self.mask_file:
            self.mask_label.setText("未找到ROI蒙版文件！请确保文件夹中包含mask.nii.gz文件。")
            return
        
        # 加载特征文件
        for f in files:
            if (f.endswith('.nii.gz') or f.endswith('.nii')) and 'mask' not in f.lower() and 'roi' not in f.lower():
                self.feature_files.append(os.path.join(self.folder_path, f))
                # 提取文件名作为特征名称
                feature_name = os.path.basename(f)
                self.feature_combo.addItem(feature_name, os.path.join(self.folder_path, f))
        
        # 如果有特征文件，加载第一个
        if self.feature_files:
            self.feature_combo.setCurrentIndex(0)
    
    def feature_changed(self, index):
        """特征选择改变时调用"""
        if index >= 0:
            self.current_feature = self.feature_combo.itemData(index)
            self.visualize_3d_feature()
            self.visualize_reference_examples()
    
    def opacity_changed(self, value):
        """透明度选择改变时调用"""
        if self.current_feature:
            self.visualize_3d_feature()
            self.visualize_reference_examples()
    
    def visualize_3d_feature(self):
        """3D可视化当前选择的特征"""
        if not self.current_feature or not self.mask_file:
            return
        
        # 获取当前参数
        opacity = float(self.opacity_combo.currentText())
        
        # 清空当前场景
        self.vedo_plt.clear()
        
        try:
            # 读取nii图像和ROI掩码
            img = sitk.ReadImage(self.current_feature)
            img_array = sitk.GetArrayFromImage(img)
            
            roi_mask = sitk.ReadImage(self.mask_file)
            roi_mask_array = sitk.GetArrayFromImage(roi_mask).astype(bool)
            
            # 获取spacing信息 (x,y,z) -> (z,y,x)
            spacing = img.GetSpacing()[::-1]
            
            # 只对ROI区域内的值进行clip操作
            processed_array = img_array.copy()
            processed_array[roi_mask_array] = np.clip(processed_array[roi_mask_array], 0.02, 1)
            
            # 创建自定义颜色映射
            colors = ['#4B0082', '#0000FF', '#00CED1', '#008000', '#FFD700', '#FFA500', '#FF4500']
            
            # 创建体积，使用处理后的数据
            vol = Volume(processed_array, spacing=spacing)
            
            # 设置颜色映射和透明度
            vol.cmap(colors)
            vol.alpha([(0, 0), (1e-7, opacity), (1, opacity)])
            vol.alpha_unit(opacity)
            vol.mode(1)
            
            # 添加到场景
            self.vedo_plt.add(vol)
            
            # 计算实际的物理尺寸
            physical_size = np.array(img_array.shape) * np.array(spacing)
            max_dim = max(physical_size)
            
            # 设置固定的等角视图
            camera_pos = np.array((1, 1, 1)) * max_dim
            self.vedo_plt.camera.SetPosition(*camera_pos)
            self.vedo_plt.camera.SetViewUp(1, 0, 0)
            
            # 添加坐标轴（使用物理尺寸，注意轴的顺序）
            ax = Axes(
                xrange=(0, physical_size[0]),  # Z轴
                yrange=(0, physical_size[1]),  # Y轴
                zrange=(0, physical_size[2]),  # X轴
                xtitle='Z (mm)', 
                ytitle='Y (mm)', 
                ztitle='X (mm)',
                xygrid=True,
                yzgrid=True,
                zxgrid=True,
                axes_linewidth=2,
                grid_linewidth=0.1,
                xygrid_color='black',
                yzgrid_color='black',
                zxgrid_color='black',
                xyalpha=0.1,
                yzalpha=0.1,
                zxalpha=0.1,
                c='black'
            )
            self.vedo_plt.add(ax)
            
            # 更新视图
            self.vedo_plt.show(interactive=False)
            self.iren.Render()
            
        except Exception as e:
            print(f"可视化错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def visualize_reference_examples(self):
        """可视化参考示例"""
        if not self.current_feature:
            return
            
        # 获取当前特征名称（不包含扩展名）
        current_feature_name = os.path.splitext(os.path.basename(self.current_feature))[0]
        if current_feature_name.endswith('.nii'):  # 处理双重扩展名
            current_feature_name = os.path.splitext(current_feature_name)[0]
            
        # 构建参考示例路径
        reference_base_path = "reference_examples"  # 参考示例的基础路径
        negative_path = os.path.join(reference_base_path, "negative")
        positive_path = os.path.join(reference_base_path, "positive")
        
        # 获取当前的透明度值
        opacity = float(self.opacity_combo.currentText())
        
        # 可视化阴性示例
        try:
            # 清空当前场景
            self.negative_plt.clear()
            
            # 检查缓存中是否存在当前特征和透明度的组合
            cache_key = (current_feature_name, opacity)
            if cache_key in self.negative_cache:
                # 从缓存中获取数据
                processed_array, spacing = self.negative_cache[cache_key]
            else:
                # 读取特征文件和mask文件
                feature_file = os.path.join(negative_path, current_feature_name + ".nii.gz")
                mask_file = os.path.join(negative_path, "mask.nii.gz")
                
                if os.path.exists(feature_file) and os.path.exists(mask_file):
                    # 读取数据
                    img = sitk.ReadImage(feature_file)
                    img_array = sitk.GetArrayFromImage(img)
                    roi_mask = sitk.ReadImage(mask_file)
                    roi_mask_array = sitk.GetArrayFromImage(roi_mask).astype(bool)
                    
                    # 使用与主显示相同的渲染逻辑
                    spacing = img.GetSpacing()[::-1]
                    processed_array = img_array.copy()
                    processed_array[roi_mask_array] = np.clip(processed_array[roi_mask_array], 0.02, 1)
                    
                    # 将处理后的数据存入缓存
                    self.negative_cache[cache_key] = (processed_array, spacing)
                else:
                    return
            
            # 使用处理后的数据创建体积
            vol = Volume(processed_array, spacing=spacing)
            vol.cmap(['#4B0082', '#0000FF', '#00CED1', '#008000', '#FFD700', '#FFA500', '#FF4500'])
            vol.alpha([(0, 0), (1e-7, opacity), (1, opacity)])
            vol.alpha_unit(opacity)
            vol.mode(1)
            
            self.negative_plt.add(vol)
            
            # 设置相同的视角
            physical_size = np.array(processed_array.shape) * np.array(spacing)
            max_dim = max(physical_size)
            self.negative_plt.camera.SetPosition(*(np.array((1, 1, 1)) * max_dim))
            self.negative_plt.camera.SetViewUp(1, 0, 0)
            
            self.negative_plt.show(interactive=False)
            self.negative_iren.Render()
        except Exception as e:
            print(f"阴性示例可视化错误: {str(e)}")
        
        # 可视化阳性示例
        try:
            # 清空当前场景
            self.positive_plt.clear()
            
            # 检查缓存中是否存在当前特征和透明度的组合
            cache_key = (current_feature_name, opacity)
            if cache_key in self.positive_cache:
                # 从缓存中获取数据
                processed_array, spacing = self.positive_cache[cache_key]
            else:
                # 读取特征文件和mask文件
                feature_file = os.path.join(positive_path, current_feature_name + ".nii.gz")
                mask_file = os.path.join(positive_path, "mask.nii.gz")
                
                if os.path.exists(feature_file) and os.path.exists(mask_file):
                    # 读取数据
                    img = sitk.ReadImage(feature_file)
                    img_array = sitk.GetArrayFromImage(img)
                    roi_mask = sitk.ReadImage(mask_file)
                    roi_mask_array = sitk.GetArrayFromImage(roi_mask).astype(bool)
                    
                    # 使用与主显示相同的渲染逻辑
                    spacing = img.GetSpacing()[::-1]
                    processed_array = img_array.copy()
                    processed_array[roi_mask_array] = np.clip(processed_array[roi_mask_array], 0.02, 1)
                    
                    # 将处理后的数据存入缓存
                    self.positive_cache[cache_key] = (processed_array, spacing)
                else:
                    return
            
            # 使用处理后的数据创建体积
            vol = Volume(processed_array, spacing=spacing)
            vol.cmap(['#4B0082', '#0000FF', '#00CED1', '#008000', '#FFD700', '#FFA500', '#FF4500'])
            vol.alpha([(0, 0), (1e-7, opacity), (1, opacity)])
            vol.alpha_unit(opacity)
            vol.mode(1)
            
            self.positive_plt.add(vol)
            
            # 设置相同的视角
            physical_size = np.array(processed_array.shape) * np.array(spacing)
            max_dim = max(physical_size)
            self.positive_plt.camera.SetPosition(*(np.array((1, 1, 1)) * max_dim))
            self.positive_plt.camera.SetViewUp(1, 0, 0)
            
            self.positive_plt.show(interactive=False)
            self.positive_iren.Render()
        except Exception as e:
            print(f"阳性示例可视化错误: {str(e)}")
    
    def closeEvent(self, event):
        """关闭窗口时的事件处理"""
        # 关闭所有渲染器和交互器
        if hasattr(self, 'vedo_plt'):
            self.vedo_plt.close()
        if hasattr(self, 'negative_plt'):
            self.negative_plt.close()
        if hasattr(self, 'positive_plt'):
            self.positive_plt.close()
        if hasattr(self, 'iren'):
            self.iren.TerminateApp()
        if hasattr(self, 'negative_iren'):
            self.negative_iren.TerminateApp()
        if hasattr(self, 'positive_iren'):
            self.positive_iren.TerminateApp()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FeatureMapVisualizer()
    window.show()
    sys.exit(app.exec_()) 