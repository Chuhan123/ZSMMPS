import os
import cv2
import numpy as np

fused_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/MobileSAM/fused_body_wing_fixpoint"
label_dir = "2d_label"

def is_colored(pixel):
    return not (pixel == [0,0,0]).all()
# 用于存储每个模型的统计数据
model_stats = {}
total = 0
correct = 0
total_intersection = 0
total_union = 0

for model_name in os.listdir(label_dir):
    label_model_dir = os.path.join(label_dir, model_name)
    fused_model_dir = os.path.join(fused_dir, model_name)
    
    if not os.path.isdir(label_model_dir) or not os.path.isdir(fused_model_dir):
        continue
    
    # 初始化当前模型的统计数据
    model_total = 0
    model_correct = 0
    model_image_count = 0
    model_intersection = 0
    model_union = 0
    
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
        
        # 计算掩码和准确率
        label_mask = np.any(label_rgb != [0,0,0], axis=-1)
        fused_mask = np.any(fused_rgb != [0,0,0], axis=-1)
        
        # 计算交集和并集
        intersection = np.logical_and(label_mask, fused_mask)
        union = np.logical_or(label_mask, fused_mask)
        
        # 计算相同像素
        same_pixels = np.all(label_rgb == fused_rgb, axis=-1)
        correct_pixels = np.sum(same_pixels & label_mask)  # 只计算标签中有颜色的区域
        
        # 计算像素数
        total_pixels = np.sum(label_mask)
        intersection_pixels = np.sum(intersection)
        union_pixels = np.sum(union)
        
        # 更新统计数据
        model_total += total_pixels
        model_correct += correct_pixels
        model_intersection += intersection_pixels
        model_union += union_pixels
        
        total += total_pixels
        correct += correct_pixels
        total_intersection += intersection_pixels
        total_union += union_pixels
        
        model_image_count += 1
        
        # 计算单张图片的指标
        acc = correct_pixels / total_pixels if total_pixels > 0 else 0
        iou = intersection_pixels / union_pixels if union_pixels > 0 else 0
    
    # 保存当前模型的统计数据
    if model_image_count > 0:
        model_acc = model_correct / model_total if model_total > 0 else 0
        model_iou = model_intersection / model_union if model_union > 0 else 0
        model_stats[model_name] = {
            'accuracy': model_acc,
            'iou': model_iou,
            'correct': model_correct,
            'total': model_total,
            'intersection': model_intersection,
            'union': model_union,
            'image_count': model_image_count
        }

# 计算并输出总体指标
overall_acc = correct / total if total > 0 else 0
overall_iou = total_intersection / total_union if total_union > 0 else 0
print(f"\n总体有色像素格准确率: {overall_acc:.4f} ({correct}/{total})")
print(f"总体交并比(IoU): {overall_iou:.4f} ({total_intersection}/{total_union})")

# 单独列出所有模型的指标
print("\n各模型指标汇总:")
for model_name, stats in model_stats.items():
    print(f"{model_name}: "
          f"准确率={stats['accuracy']:.4f} ({stats['correct']}/{stats['total']}), "
          f"IoU={stats['iou']:.4f} ({stats['intersection']}/{stats['union']}), "
          f"共{stats['image_count']}张图片")