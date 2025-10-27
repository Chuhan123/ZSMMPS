import os
import numpy as np
from pathlib import Path
def calculate_shape_IoU(pred_np, seg_np):
    shape_ious = []
    category = {}  # 用于存储每个类别的IoU
    for shape_idx in range(seg_np.shape[0]):
        part_ious = [] 
        for part in (1,2,3):  
            I = np.sum(np.logical_and(pred_np == part, seg_np == part))
            U = np.sum(np.logical_or(pred_np== part, seg_np == part))
            if U == 0:
                iou = 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
            # 将当前类别的IoU存入category字典（关键补充）
            if part not in category:
                category[part] = []
            category[part].append(iou)  # 记录每个部件的IoU
        shape_ious.append(np.mean(part_ious))
    return shape_ious, category  # 返回两个值：shape_ious和category


def calculate_metrics(pre_antenna_dir, final_seg_dir):
    """
    计算预测结果的准确率和整体IOU（区分部件）
    
    参数:
        pre_antenna_dir: 真实标签文件夹路径（包含PLY文件）
        final_seg_dir: 预测结果文件夹路径（包含PLY文件）
    """
    # 获取两个文件夹中匹配的文件名（不含后缀）
    pre_files = {Path(f).stem: f for f in os.listdir(pre_antenna_dir) 
                if os.path.isfile(os.path.join(pre_antenna_dir, f))}
    final_files = {Path(f).stem: f for f in os.listdir(final_seg_dir) 
                  if os.path.isfile(os.path.join(final_seg_dir, f))}
    matched_stems = set(pre_files.keys()) & set(final_files.keys())
    
    if not matched_stems:
        print("警告：未找到匹配的文件！")
        return
    
    print(f"找到{len(matched_stems)}个匹配文件，开始计算指标...\n")
    
    # 存储所有样本的预测/真实标签数组（用于总体IOU计算）
    all_preds = []
    all_segs = []
    # 总体准确率统计
    total_matched_all = 0
    correct_matched_all = 0
    
    # 遍历每个匹配文件
    for stem in matched_stems:
        pre_path = os.path.join(pre_antenna_dir, pre_files[stem])
        final_path = os.path.join(final_seg_dir, final_files[stem])
        
        try:
            # 1. 读取真实标签文件（pre_antenna_dir）
            pre_coords = {}  # 键：(x,y,z)坐标，值：真实标签
            with open(pre_path, 'r') as f:
                lines = f.readlines()
                # 跳过PLY文件头（直到end_header）
                start_idx = None
                for i, line in enumerate(lines):
                    if line.strip() == 'end_header':
                        start_idx = i + 1
                        break
                if start_idx is None:
                    print(f"文件{pre_path}格式错误（无end_header），跳过")
                    continue
                # 读取点坐标和标签（假设第4列是标签）
                for line in lines[start_idx:]:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue  # 跳过无效行
                    x, y, z = map(float, parts[:3])
                    true_label = int(parts[3])
                    # 坐标保留6位小数，避免浮点数精度问题
                    coord_key = (round(x, 6), round(y, 6), round(z, 6))
                    pre_coords[coord_key] = true_label
            
            # 2. 读取预测标签文件（final_seg_dir）并对比
            pred_labels = []  # 当前文件的预测标签列表
            true_labels = []  # 当前文件的真实标签列表
            total_matched = 0  # 坐标匹配的点总数
            correct_matched = 0  # 标签正确的点数量
            
            with open(final_path, 'r') as f:
                lines = f.readlines()
                start_idx = None
                for i, line in enumerate(lines):
                    if line.strip() == 'end_header':
                        start_idx = i + 1
                        break
                if start_idx is None:
                    print(f"文件{final_path}格式错误（无end_header），跳过")
                    continue
                # 读取点坐标和预测标签
                for line in lines[start_idx:]:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue  # 跳过无效行
                    x, y, z = map(float, parts[:3])
                    pred_label = int(parts[3])
                    coord_key = (round(x, 6), round(y, 6), round(z, 6))
                    
                    # 仅处理坐标匹配的点
                    if coord_key in pre_coords:
                        total_matched += 1
                        true_label = pre_coords[coord_key]
                        pred_labels.append(pred_label)
                        true_labels.append(true_label)
                        if pred_label == true_label:
                            correct_matched += 1
            
            # 3. 输出当前文件的指标
            if total_matched == 0:
                print(f"文件 {stem}：无坐标匹配的点，跳过")
                continue
            
            # 计算准确率
            accuracy = (correct_matched / total_matched) * 100
            print(f"文件 {stem}：")
            print(f"  匹配点总数：{total_matched}")
            print(f"  正确标签数：{correct_matched}")
            print(f"  准确率：{accuracy:.2f}%")
            
            # 准备IOU计算的输入
            pred_np = np.array(pred_labels)
            seg_np = np.array(true_labels)
            # 计算当前文件的整体IOU
            shape_ious, _ = calculate_shape_IoU(pred_np, seg_np)
            print(f"平均IoU：{shape_ious[0]:.4f}\n")
            
            # 收集数据用于总体统计
            all_preds.append(pred_np)
            all_segs.append(seg_np)
            total_matched_all += total_matched
            correct_matched_all += correct_matched
        
        except Exception as e:
            print(f"处理文件 {stem} 时出错：{str(e)}\n")
    
    # 4. 计算并输出总体指标
    if total_matched_all == 0:
        print("所有文件均无有效匹配点，无法计算总体指标")
        return
    
    # 总体准确率
    overall_accuracy = (correct_matched_all / total_matched_all) * 100
    print("="*50)
    print(f"总体统计：")
    print(f"  总匹配点数量：{total_matched_all}")
    print(f"  总正确标签数量：{correct_matched_all}")
    print(f"  总体准确率：{overall_accuracy:.2f}%")
    
    # 总体IOU（合并所有样本计算）
    if all_preds and all_segs:
        all_preds_np = np.concatenate(all_preds)
        all_segs_np = np.concatenate(all_segs)
        _, all_category = calculate_shape_IoU(all_preds_np, all_segs_np)
        
        # 计算每个类别的平均IoU
        print("\n每个类别的平均IoU：")
        for part, ious in all_category.items():
            print(f"  类别 {part}：{np.mean(ious):.4f}")
        
        # 计算所有类别的平均IoU
        all_ious = []
        for ious in all_category.values():
            all_ious.extend(ious)
        overall_mean_iou = np.mean(all_ious)
        print(f"\n  所有样本的平均IoU：{overall_mean_iou:.4f}")

if __name__ == "__main__":
    pre_antenna_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/data/nasa/test/point2048"  # 真实标签文件夹
    final_seg_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/ZSMMPS/pre_antenna"  # 预测结果文件夹
    calculate_metrics(pre_antenna_dir, final_seg_dir)
