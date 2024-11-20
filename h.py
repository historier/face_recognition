import os
import cv2
import numpy as np
from PIL import Image
import time

# 获取所有图像文件的路径
def get_all_image_paths(dirpath, *suffixes):
    path_array = []
    for root, _, files in os.walk(dirpath):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in suffixes:
                filepath = os.path.join(root, filename)
                path_array.append(filepath)
    return path_array

# 读取图像处理中文路径
def read_image_with_chinese_path(image_path):
    pil_image = Image.open(image_path).convert('L')  # 灰度模式
    open_cv_image = np.array(pil_image)
    return open_cv_image

# 将非 JPG 格式的图片转换为 JPG
def convert_to_jpg(image_path):
    img = Image.open(image_path)
    if img.format != 'JPEG':
        # 生成新的 JPG 路径
        new_path = os.path.splitext(image_path)[0] + ".jpg"
        img = img.convert('RGB')
        img.save(new_path, "JPEG")
        return new_path
    return image_path

# 读取图片并检测人脸，然后将人脸区域保存到目标路径
def detect_and_save_faces(source_path, target_path, *suffixes):
    try:
        # 获取所有图片路径
        image_paths = get_all_image_paths(source_path, *suffixes)
        count = 0
        
        # 加载人脸级联分类器
        cascade_path = "F:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml"  # 替换为实际路径
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        for image_path in image_paths:
            # 转换为 JPG 格式（如果不是 JPG）
            image_path = convert_to_jpg(image_path)
            
            # 读取图像并处理中文路径
            img = read_image_with_chinese_path(image_path)
            
            # 人脸检测
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(128, 128))
            
            # 获取文件夹名作为标签
            person_name = os.path.basename(os.path.dirname(image_path))
            
            for (x, y, w, h) in faces:
                # 确保人脸区域足够大
                if w >= 128 and h >= 128:
                    face_img = img[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (200, 200))  # 调整为200x200大小

                    # 构造目标文件路径
                    filename = os.path.basename(image_path)
                    save_folder = os.path.join(target_path, person_name)
                    os.makedirs(save_folder, exist_ok=True)
                    save_path = os.path.join(save_folder, filename)

                    # 保存裁剪后的图片
                    cv2.imwrite(save_path, face_img)
                    count += 1

    except Exception as e:
        print(f"Error: {e}")

    else:
        print(f'Found and saved {count} faces to destination folder: {target_path}')

if __name__ == '__main__':
    start_time = time.time()
    # 定义原始数据文件夹路径和人脸裁剪结果存储路径
    source_path = r'"F:\\kpop"'  # 原始数据文件夹路径
    target_path = r'F:\\kpop1'  # 人脸裁剪结果存储路径
    
    # 检测人脸并保存裁剪结果
    detect_and_save_faces(source_path, target_path, '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    
    end_time = time.time()
    print(f"程序运行时间：{end_time - start_time:.2f}秒")
    print(f"人脸裁剪结果已保存到文件夹: {target_path}")
