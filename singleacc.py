import cv2
import numpy as np
import os

def compare_non_black_pixels(image_path1, image_path2):
    """
    只在label图的非黑像素区域比较两张图片的像素值，返回准确率
    
    参数:
    image_path1 (str): 第一张图片(fuse_body_wing)的路径
    image_path2 (str): 第二张图片(label_2d)的路径
    
    返回:
    float: 非黑像素区域的准确率 (0-1之间)
    """
    # 读取图片
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    # 检查图片是否成功加载
    if img1 is None:
        raise FileNotFoundError(f"无法加载图片: {image_path1}")
    if img2 is None:
        raise FileNotFoundError(f"无法加载图片: {image_path2}")
    
    # 确保两张图片尺寸相同
    if img1.shape != img2.shape:
        raise ValueError("两张图片尺寸不一致")
    
    # 创建label图的非黑像素掩码
    non_black_mask = np.any(img2 != [0, 0, 0], axis=-1)
    
    # 计算label图中非黑像素的总数
    total_non_black_pixels = np.sum(non_black_mask)
    
    # 提取非黑像素位置的像素值进行比较
    pixels1 = img1[non_black_mask]
    pixels2 = img2[non_black_mask]
    
    # 逐像素比较RGB值
    matching_pixels = np.all(pixels1 == pixels2, axis=1)
    
    # 计算匹配的像素数
    correct_matches = np.sum(matching_pixels)
    
    # 计算准确率
    accuracy = correct_matches / total_non_black_pixels
    
    return accuracy

# 示例使用
if __name__ == "__main__":
    image_path1 = "fuse_body_wing/Loral-1300Com-main/frame_0001.png"
    image_path2 = "label_2d/Loral-1300Com-main/frame_0001.png"
    
    try:
        accuracy = compare_non_black_pixels(image_path1, image_path2)
        print(f"label图非黑像素区域的准确率: {accuracy:.2%}")
    except Exception as e:
        print(f"错误: {e}")