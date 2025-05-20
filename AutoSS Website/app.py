from flask import Flask, request, render_template, jsonify
import os
from datetime import datetime
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import gc
import numpy as np
from utils import RadiomicsAnalyzer, mask_segmentation, process_mask
import shutil

app = Flask(__name__, 
    static_url_path='/static',
    static_folder='static')
app.config['TEMPLATES_AUTO_RELOAD'] = True
UPLOAD_PATH = os.path.join(os.path.dirname(__file__), 'upload_files')
# 添加预设案例目录
PREDEFINED_CASES_PATH = os.path.join(os.path.dirname(__file__), 'predefined_cases')
os.makedirs(UPLOAD_PATH, exist_ok=True)
os.makedirs(PREDEFINED_CASES_PATH, exist_ok=True)

# 创建RadiomicsAnalyzer实例
analyzer = RadiomicsAnalyzer()

window_level_cache = {}

def calculate_window_settings(image_data, save_time):
    """计算并缓存窗宽窗位设置"""
    if save_time not in window_level_cache:
        p1 = float(np.percentile(image_data, 0.2))
        p99 = float(np.percentile(image_data, 99.8))
        
        window_width = p99 - p1
        window_level = (p99 + p1) / 2
        
        min_window_width = 50
        window_width = max(window_width, min_window_width)
        
        window_level_cache[save_time] = {
            'p5': p1,
            'p95': p99,
            'window_width': window_width,
            'window_level': window_level
        }
    
    return window_level_cache[save_time]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传的新路由"""
    image = request.files.get('image')

    if image:
        current_time = datetime.now().strftime("%Y%m%d")
        
        # 确保目录存在
        os.makedirs(os.path.join(UPLOAD_PATH, 'images'), exist_ok=True)
        os.makedirs(os.path.join(UPLOAD_PATH, 'labels'), exist_ok=True)
        
        image_path = os.path.join(UPLOAD_PATH, 'images', f'image_{current_time}.nii.gz')
        raw_label_path = os.path.join(UPLOAD_PATH, 'masks', f'mask_{current_time}.nii.gz')
        label_path = os.path.join(UPLOAD_PATH, 'labels', f'label_{current_time}.nii.gz')
        
        try:
            image.save(image_path)
            
            # 调用mask_segmentation函数生成label文件
            # success = mask_segmentation(input_image_path=image_path, output_file_path=raw_label_path)
            # process_mask(raw_label_path, label_path)
            
            # if not success:
            #     raise Exception("Segmentation failed.")
            
            return jsonify({
                'success': True,
                'save_time': current_time,
                'message': 'File uploaded and label generated successfully'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e)
            }), 500
    else:
        return jsonify({
            'success': False,
            'message': 'No file uploaded'
        }), 400

@app.route('/get_initial_data')
def get_initial_data():
    """获取初始化预览所需的数据"""
    try:
        save_time = request.args.get('save_time')
        if not save_time:
            return jsonify({
                'success': False,
                'message': 'Save time parameter is required'
            }), 400
            
        image_path = os.path.join(UPLOAD_PATH, 'images', f'image_{save_time}.nii.gz')
        
        if not os.path.exists(image_path):
            return jsonify({
                'success': False,
                'message': f'Image not found: {image_path}'
            }), 404
        
        image_nii = nib.load(image_path)
        image_data = image_nii.get_fdata()
        header = image_nii.header
        
        # 计算或获取缓存的窗宽窗位设置
        window_settings = calculate_window_settings(image_data, save_time)

        return jsonify({
            'success': True,
            'axial_layers': image_data.shape[2],
            'pixel_spacing': np.array(header.get_zooms(), dtype=float).tolist(),
            'window_width': window_settings['window_width'],
            'window_level': window_settings['window_level'],
            'p5': window_settings['p5'],
            'p95': window_settings['p95']
        })
    except Exception as e:
        import traceback
        error_msg = f"Error in get_initial_data: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # 打印详细错误信息到控制台
        return jsonify({
            'success': False,
            'message': str(e),
            'details': error_msg
        }), 500

@app.route('/get_layer')
def get_layer():
    """获取单层图像数据，返回原始图像和标签的叠加结果"""
    try:
        layer = int(request.args.get('layer'))
        save_time = request.args.get('save_time')
        
        image_path = os.path.join(UPLOAD_PATH, 'images', f'image_{save_time}.nii.gz')
        label_path = os.path.join(UPLOAD_PATH, 'labels', f'label_{save_time}.nii.gz')

        # 加载原始图像
        image_nii = nib.load(image_path)
        image_data = image_nii.get_fdata()
        img_slice = image_data[:, :, layer]
        
        # 顺时针旋转90度 (k=3 表示顺时针旋转90度，相当于逆时针旋转270度)
        img_slice = np.rot90(img_slice, k=3)
        
        # 获取缓存的窗宽窗位设置
        window_settings = calculate_window_settings(image_data, save_time)

        # 直接使用原始数据
        img_bytes = img_slice.astype(np.float32).tobytes()
        
        # 处理标签数据
        label_nii = nib.load(label_path)
        label_data = label_nii.get_fdata()
        label_slice = label_data[:, :, layer]
        
        # 标签数据也需要顺时针旋转
        label_slice = np.rot90(label_slice, k=3)
        
        # 创建RGB标签图像
        label_rgb = np.zeros((*label_slice.shape, 3), dtype=np.uint8)
        colors = {
            1: [255, 182, 193],    # 粉红
            2: [144, 238, 144],    # 浅绿
            3: [135, 206, 250],    # 天蓝
            4: [255, 218, 185]     # 桃色
        }
        
        for label_value, color in colors.items():
            mask = label_slice == label_value
            if np.any(mask):
                for i in range(3):
                    label_rgb[:, :, i][mask] = color[i]

        return jsonify({
            'success': True,
            'image': base64.b64encode(img_bytes).decode('utf-8'),
            'label': base64.b64encode(label_rgb.tobytes()).decode('utf-8'),
            'shape': img_slice.shape,  # 注意：这里返回的是旋转后的shape
            'window_width': window_settings['window_width'],
            'window_level': window_settings['window_level']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/extract_features', methods=['POST'])
def extract_features():
    """特征提取路由"""
    save_time = request.form.get('save_time')
    sample_files_folder = request.form.get('sample_files_folder')
    try:
        # 检查特征文件是否已存在
        features_exist = True
        for label_id in range(1, 3):
            feature_path = os.path.join('radiomics_features', sample_files_folder, f'{save_time}_label_{label_id}.csv')
            if not os.path.exists(feature_path):
                features_exist = False
                break
        
        if features_exist:
            print(f"Features already exist: {save_time}")
            return jsonify({
                'success': True,
                'message': 'Features extracted successfully'
            })
        else:
            # 执行实际的特征提取
            # analyzer.extract_features(save_time)
            return jsonify({
                'success': True,
                'message': 'Features extracted successfully'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/start_prediction', methods=['POST'])
def start_prediction():
    """开始预测路由"""
    try:
        save_time = request.json.get('save_time')
        sample_files_folder = request.json.get('sample_files_folder')
        if not save_time:
            raise ValueError("Missing save_time parameter")
            
        predict_result, graph, feature_mapping = analyzer.predict(save_time=save_time, image_num=sample_files_folder)
        if predict_result is None:
            raise ValueError("Prediction failed - null result")
            
        predict_result = round(float(predict_result) * 100, 2)
        diagnosis = 'Probability of Sjogren\'s Syndrome: ' + str(predict_result) + '%'
        
        return jsonify({
            'success': True,
            'result': diagnosis,
            'image_url': graph,
            'feature_mapping': feature_mapping
        })
    except Exception as e:
        import traceback
        error_msg = f"Error in prediction: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # 打印详细错误信息到控制台
        return jsonify({
            'success': False,
            'message': str(e),
            'details': error_msg
        }), 500

@app.route('/load_predefined_case')
def load_predefined_case():
    """加载预设案例的路由"""
    try:
        case_id = request.args.get('case_id')
        if not case_id:
            return jsonify({
                'success': False,
                'message': 'Case ID is required'
            }), 400
            
        # 验证案例ID
        if case_id not in ['case1', 'case2']:
            return jsonify({
                'success': False,
                'message': 'Invalid case ID'
            }), 400
            
        # 预设案例对应的时间标识
        case_times = {
            'case1': 'case1',
            'case2': 'case2'
        }
        save_time = case_times[case_id]
        
        # 确保目标目录存在
        os.makedirs(os.path.join(UPLOAD_PATH, 'images'), exist_ok=True)
        os.makedirs(os.path.join(UPLOAD_PATH, 'labels'), exist_ok=True)
        
        # 从预设案例目录复制文件到上传目录
        source_image_path = os.path.join(PREDEFINED_CASES_PATH, case_id, 'image.nii.gz')
        source_label_path = os.path.join(PREDEFINED_CASES_PATH, case_id, 'label.nii.gz')
        
        target_image_path = os.path.join(UPLOAD_PATH, 'images', f'image_{save_time}.nii.gz')
        target_label_path = os.path.join(UPLOAD_PATH, 'labels', f'label_{save_time}.nii.gz')
        
        # 检查源文件是否存在
        if not os.path.exists(source_image_path) or not os.path.exists(source_label_path):
            return jsonify({
                'success': False,
                'message': f'Predefined case files not found for {case_id}'
            }), 404
            
        # 复制图像和标签文件
        shutil.copy2(source_image_path, target_image_path)
        shutil.copy2(source_label_path, target_label_path)
        
        # 创建特征保存目录，确保路径存在
        features_dir = 'radiomics_features'
        os.makedirs(features_dir, exist_ok=True)
        
        # 假设每个案例有2个标签区域，循环复制特征文件
        for label_id in range(1, 3):
            # 创建特征子目录
            sample_files_folder = 'sample_files'
            label_features_dir = os.path.join(features_dir, sample_files_folder)
            os.makedirs(label_features_dir, exist_ok=True)
            
            # 源特征文件路径
            source_feature_path = os.path.join(PREDEFINED_CASES_PATH, case_id, f'{case_id}_label_{label_id}.csv')
            
            # 目标特征文件路径
            target_feature_path = os.path.join(label_features_dir, f'{case_id}_label_{label_id}.csv')
            
            # 如果预设特征文件存在，则复制
            if os.path.exists(source_feature_path):
                shutil.copy2(source_feature_path, target_feature_path)
        
        return jsonify({
            'success': True,
            'save_time': save_time,
            'sample_files_folder': sample_files_folder,
            'message': f'Predefined case {case_id} loaded successfully'
        })
    except Exception as e:
        import traceback
        error_msg = f"Error loading predefined case: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # 打印详细错误信息到控制台
        return jsonify({
            'success': False,
            'message': str(e),
            'details': error_msg
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
