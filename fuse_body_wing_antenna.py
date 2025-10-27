import os
import cv2
import numpy as np

# 定义蓝色像素的HSV范围 (调整范围以适应实际蓝色)
lower_blue = np.array([100, 150, 50])    # 最小HSV值 (色相, 饱和度, 明度)
upper_blue = np.array([140, 255, 255])   # 最大HSV值

# 文件夹路径
antenna_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/2d_antenna"
fuse_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/ZSMMPS/补充"
output_dir = "fused_bwa_fixpoint"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 遍历antenna目录结构
for model_name in os.listdir(antenna_dir):
    model_path_ant = os.path.join(antenna_dir, model_name)
    model_path_fuse = os.path.join(fuse_dir, model_name)
    model_output = os.path.join(output_dir, model_name)
    
    if not os.path.isdir(model_path_ant): 
        continue
    
    # 创建模型子目录
    os.makedirs(model_output, exist_ok=True)
    
    # 处理每个视图图像
    for view_file in os.listdir(model_path_ant):
        ant_path = os.path.join(model_path_ant, view_file)
        fuse_path = os.path.join(model_path_fuse, view_file)
        out_path = os.path.join(model_output, view_file)
        
        if not os.path.exists(fuse_path):
            print(f"警告: 缺失对应文件 {fuse_path}")
            continue
            
        # 读取图像
        img_ant = cv2.imread(ant_path)
        img_fuse = cv2.imread(fuse_path)
        
        if img_ant is None or img_fuse is None:
            print(f"错误: 读取图像失败 {ant_path} 或 {fuse_path}")
            continue
            
        # 确保图像尺寸相同
        if img_ant.shape != img_fuse.shape:
            img_fuse = cv2.resize(img_fuse, (img_ant.shape[1], img_ant.shape[0]))
        
        # 转换到HSV空间检测蓝色
        hsv_ant = cv2.cvtColor(img_ant, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_ant, lower_blue, upper_blue)
        
        # 创建蓝色像素替换层
        blue_layer = np.zeros_like(img_fuse)
        blue_layer[blue_mask > 0] = (255, 0, 0)  # BGR格式的蓝色
        
        # 合并到目标图像
        result = img_fuse.copy()
        result[blue_mask > 0] = blue_layer[blue_mask > 0]
        
        # 保存结果
        cv2.imwrite(out_path, result)
        #print(f"处理完成: {model_name}/{view_file}")

print("所有图像处理完毕！输出目录:", output_dir)