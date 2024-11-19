import cv2
import numpy as np
from PIL import Image
import os


CADES_PATH = "C:\\Users\\LEGION\\AppData\\Roaming\\Python\\Python39\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml"

#该函数可取代cv2.imread
def cv_imread(file_path: str):
    """
    使用PIL读取一张图片并转换成OpenCV格式, 从而支持读取中文路径的图片。
    """

    # 使用PIL打开图片文件
    img = Image.open(file_path)
    # 将PIL图像转换为NumPy数组( RGB格式 )
    cv_img_rgb = np.array(img)
    # 使用OpenCV将RGB格式的NumPy数组转换为BGR格式
    cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)
    # 返回转换后的BGR格式的NumPy数组
    return cv_img_bgr

#该函数可取代cv2.imwrite
def cv_imwrite(file_path: str, img: cv2.typing.MatLike):
    # 从文件路径中提取文件扩展名
    extension = os.path.splitext(file_path)[1]

    # 使用OpenCV的imencode函数对图像进行编码
    # 注意: extension包括点（.）, 例如 '.jpg' 而不是 'jpg'
    # 编码后的图像数据将根据文件扩展名自动选择正确的格式
    encoded_img = cv2.imencode(extension, img)[1]

    # 将编码后的图像数据写入文件
    encoded_img.tofile(file_path)


def face_detect(img_path):
    color = (0, 255, 0)
    img_bgr = cv_imread(img_path)
    classifier = cv2.CascadeClassifier(CADES_PATH)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    facerects = classifier.detectMultiScale(img_gray)
    if len(facerects) > 0:
        for rect in facerects:
            x, y, w, h = rect
            if w > 200:
                cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
    cv_imwrite('1.jpg', img_bgr)


if __name__ == '__main__':
    face_detect('face_data/张婧仪/1.jpg')