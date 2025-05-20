import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import time


def extract_radiomics_features(save_time):
    # return time.sleep(0.5)
    # 设置路径
    image_file = os.path.join(r"upload_files/images", f'image_{save_time}.nii.gz')
    label_file = os.path.join(r"upload_files/labels", f'label_{save_time}.nii.gz')

    # 设置CSV文件路径
    csv_files = {
        1: f"radiomics_features/{save_time}_label_1.csv",
        2: f"radiomics_features/{save_time}_label_2.csv",
        3: f"radiomics_features/{save_time}_label_3.csv",
        4: f"radiomics_features/{save_time}_label_4.csv"
    }

    # 创建特征提取器（仅创建一次）
    _settings = {'geometryTolerance': 10}
    extractor = featureextractor.RadiomicsFeatureExtractor(**_settings)
    # 滤波器设置
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName('Original')
    extractor.enableImageTypeByName('Square')
    extractor.enableImageTypeByName('SquareRoot')
    extractor.enableImageTypeByName('Logarithm')

    # 特征设置
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['Kurtosis'],
                                   shape=['SurfaceArea'],
                                   glrlm=['LongRunHighGrayLevelEmphasis', 'RunEntropy'],
                                   gldm=['DependenceNonUniformity', 'GrayLevelNonUniformity'],
                                   glcm=['Contrast', 'ClusterShade'],
                                   glszm=['GrayLevelVariance'])

    # 读取原始NII文件
    image_nii = sitk.ReadImage(image_file)
    label_nii = sitk.ReadImage(label_file)

    # 为每个标签提取特征
    for label, csv_file in csv_files.items():
        # 重置设置
        extractor.settings['label'] = label
        feature = extractor.execute(image_nii, label_nii)
        result = pd.DataFrame([feature])

        # 将特征追加到CSV文件中
        result.to_csv(csv_files[label], mode='a', index=False, header=not os.path.exists(csv_files[label]))

    # 删除上传的文件
    os.remove(image_file)
    os.remove(label_file)
