import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm  # 用于显示进度条，需要安装: pip install tqdm

# 定义蓝色在HSV中的范围（可根据实际情况调整）
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

# 遍历antenna文件夹下的所有模型文件夹及图片
antenna_dir = Path("fused_body_wing_antenna")

# 统计总文件数用于进度显示
total_files = 0
for model_dir in antenna_dir.iterdir():
    if model_dir.is_dir():
        total_files += len([img_path for img_path in model_dir.glob("*.*") if img_path.suffix == ".png"])

print(f"找到 {total_files} 个PNG文件需要处理")
processed_files = 0

# 使用tqdm显示进度条
for model_dir in tqdm(list(antenna_dir.iterdir()), desc="处理模型文件夹"):
    if model_dir.is_dir():
        # 新建processed子文件夹用于保存处理后的图片，若已存在则不用重复创建
        processed_dir = model_dir
        processed_dir.mkdir(exist_ok=True)
        
        # 获取当前模型下的所有PNG文件
        img_paths = list(model_dir.glob("*.*"))
        img_paths = [img_path for img_path in img_paths if img_path.suffix == ".png"]
        
        for img_path in tqdm(img_paths, desc=f"处理 {model_dir.name}", leave=False):
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                continue  # 若读取失败，跳过该图片
            
            # 转换为HSV色彩空间
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 提取蓝色区域掩码
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # 寻找蓝色区域的轮廓
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 在蓝色轮廓区域内，将非蓝色像素转为蓝色
            for cnt in contours:
                # 创建轮廓的掩码
                cnt_mask = np.zeros_like(blue_mask)
                cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
                
                # 在轮廓掩码范围内，将非蓝色的像素（即hsv中不在蓝色范围的）转为蓝色
                for y in range(img.shape[0]):
                    for x in range(img.shape[1]):
                        if cnt_mask[y, x] == 255:  # 在蓝色轮廓区域内
                            pixel_hsv = hsv[y, x]
                            if not (lower_blue[0] <= pixel_hsv[0] <= upper_blue[0] and
                                    lower_blue[1] <= pixel_hsv[1] <= upper_blue[1] and
                                    lower_blue[2] <= pixel_hsv[2] <= upper_blue[2]):
                                # 将该像素设置为蓝色
                                img[y, x] = [255, 0, 0]
            
            # 保存处理后的图片
            processed_img_path = processed_dir / img_path.name
            cv2.imwrite(str(processed_img_path), img)
            
            # 更新处理进度
            processed_files += 1
            if processed_files % 10 == 0 or processed_files == total_files:
                progress = processed_files / total_files * 100
                print(f"已处理: {processed_files}/{total_files} ({progress:.1f}%)")

print("所有图片处理完成!")