import os
import cv2
import numpy as np
from mtcnn import MTCNN


def read_image_with_chinese_path(image_path):
    """读取包含中文路径的图像文件"""
    img_array = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image


def save_image_with_chinese_path(image, save_path):
    """保存图像到包含中文路径的文件"""
    cv2.imencode('.jpg', image)[1].tofile(save_path)


def process_and_save_faces_mtcnn(source_folder, target_folder, img_size=(200, 200),
                                 suffix=('.jpg', '.jpeg', '.png', '.bmp')):
    """
    使用 MTCNN 对每个人文件夹中的照片进行人脸检测、裁剪、灰度化、统一尺寸，并将处理后图像保存到新的文件夹中。

    Args:
        source_folder (str): 包含150个人照片的原始总文件夹路径
        target_folder (str): 保存处理后照片的目标总文件夹路径
        img_size (tuple): 裁剪后的图像尺寸，默认200x200像素
        suffix (tuple): 需要处理的图片文件格式后缀，默认包括常见格式
    """
    if not os.path.exists(source_folder):
        print("源文件夹路径不存在:", source_folder)
        return

    # 创建目标文件夹，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 初始化 MTCNN 检测器
    detector = MTCNN()

    # 遍历所有子文件夹
    for person_folder in os.listdir(source_folder):
        person_source_path = os.path.join(source_folder, person_folder)

        # 确保只处理文件夹
        if not os.path.isdir(person_source_path):
            continue

        # 在目标文件夹中创建对应的子文件夹
        person_target_path = os.path.join(target_folder, person_folder)
        if not os.path.exists(person_target_path):
            os.makedirs(person_target_path)

        # 处理该文件夹中的所有图片文件
        for index, filename in enumerate(os.listdir(person_source_path)):
            if filename.lower().endswith(suffix):
                image_path = os.path.join(person_source_path, filename)

                # 尝试读取图片
                img = read_image_with_chinese_path(image_path)
                if img is None:
                    print(f"无法读取文件: {image_path}")
                    continue

                # 使用 MTCNN 检测人脸
                faces = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                # 如果找到人脸，进行裁剪
                if len(faces) > 0:
                    # 选择置信度最高的人脸
                    face = max(faces, key=lambda f: f['confidence'])
                    x, y, w, h = face['box']  # 获取人脸区域的坐标
                    face_img = img[y:y + h, x:x + w]  # 裁剪人脸区域
                    resized_face = cv2.resize(face_img, img_size)  # 调整到统一尺寸
                    gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)  # 转为灰度图

                    # 保存到目标文件夹，按数字编号命名
                    new_filename = f"{index + 1:04d}.jpg"  # 生成 0001.jpg, 0002.jpg 等
                    new_image_path = os.path.join(person_target_path, new_filename)

                    # 保存处理后的图片
                    save_image_with_chinese_path(gray_face, new_image_path)
                    print(f"已保存裁剪并重命名的图片: {new_image_path}")

    print("所有图片已处理并保存到新文件夹完成。")


# 示例使用
source_folder = r'face_data'   # 原始文件夹路径
target_folder = r'face_new'  # 新的目标文件夹路径
process_and_save_faces_mtcnn(source_folder, target_folder)
