import os
import cv2
import numpy as np

fused_dir = "fused_bwa_fixpoint"
label_dir = "2d_label"

# 用于存储每个模型的统计数据
model_stats = {}
total = 0
total = 0
correct = 0

for model_name in os.listdir(label_dir):
    label_model_dir = os.path.join(label_dir, model_name)
    fused_model_dir = os.path.join(fused_dir, model_name)
    
    if not os.path.isdir(label_model_dir) or not os.path.isdir(fused_model_dir):
        continue
    
    # 初始化当前模型的统计数据
    model_total = 0
    model_correct = 0
    model_image_count = 0
    
    for fname in os.listdir(label_model_dir):
        if not (fname.endswith('.png') or fname.endswith('.jpg')):
            continue
            
        label_path = os.path.join(label_model_dir, fname)
        # 统一融合图像的扩展名
        fused_path = os.path.join(fused_model_dir, fname.replace('.jpg', '.png').replace('.jpeg', '.png'))
        
        if not os.path.exists(fused_path):
            continue
            
        label_img = cv2.imread(label_path)
        fused_img = cv2.imread(fused_path)
        
        if label_img is None or fused_img is None or label_img.shape != fused_img.shape:
            print(f"跳过尺寸不符: {label_path}")
            continue
        
        # 转换颜色空间
        label_rgb = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
        fused_rgb = cv2.cvtColor(fused_img, cv2.COLOR_BGR2RGB)
        
        # 计算所有像素的总数和正确匹配数（包括黑色像素）
        # 获取图像总像素数 (高度 × 宽度)
        total_pixels = label_rgb.shape[0] * label_rgb.shape[1]
        # 计算所有像素中匹配的数量
        correct_pixels = np.sum(np.all(label_rgb == fused_rgb, axis=-1))
        
        # 更新统计数据
        model_total += total_pixels
        model_correct += correct_pixels
        total += total_pixels
        correct += correct_pixels
        model_image_count += 1
        
        # 计算单张图片的准确率
        acc = correct_pixels / total_pixels if total_pixels > 0 else 0
    
    # 保存当前模型的统计数据
    if model_image_count > 0:
        model_acc = model_correct / model_total if model_total > 0 else 0
        model_stats[model_name] = {
            'accuracy': model_acc,
            'correct': model_correct,
            'total': model_total,
            'image_count': model_image_count
        }

# 计算并输出总体准确率
overall_acc = correct / total if total > 0 else 0
print(f"\n总体所有像素准确率: {overall_acc:.4f} ({correct}/{total})")

# 单独列出所有模型的平均准确率
print("\n各模型平均准确率汇总:")
for model_name, stats in model_stats.items():
    print(f"{model_name}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']}) 共{stats['image_count']}张图片")
