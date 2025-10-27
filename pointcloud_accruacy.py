import os
from pathlib import Path

def calculate_label_accuracy(pre_antenna_dir, final_seg_dir):
    """
    计算final_seg文件夹中与pre_antenna_dir中同名文件的坐标相同点的标签准确率，包括总体准确率
    
    参数:
        pre_antenna_dir: 参考标签文件夹路径（data/nasa/test/point2048）
        final_seg_dir: 待评估文件夹路径（final_seg）
    """
    # 获取两个文件夹中所有文件的名称（不含后缀）
    pre_files = {Path(f).stem: f for f in os.listdir(pre_antenna_dir) 
                if os.path.isfile(os.path.join(pre_antenna_dir, f))}
    final_files = {Path(f).stem: f for f in os.listdir(final_seg_dir) 
                  if os.path.isfile(os.path.join(final_seg_dir, f))}
    
    # 仅处理名称匹配的文件
    matched_stems = set(pre_files.keys()) & set(final_files.keys())
    if not matched_stems:
        print("警告：两个文件夹中没有名称匹配的文件！")
        return
    
    print(f"找到{len(matched_stems)}个匹配文件，开始计算准确率...\n")
    
    # 新增：用于统计总体的变量
    total_matched_all = 0  # 所有文件的总匹配点数
    correct_matched_all = 0  # 所有文件的总正确匹配点数
    
    # 遍历所有匹配文件
    for stem in matched_stems:
        pre_path = os.path.join(pre_antenna_dir, pre_files[stem])
        final_path = os.path.join(final_seg_dir, final_files[stem])
        
        try:
            # 1. 读取pre_antenna文件的坐标和标签（作为参考）
            pre_coords = {}  # 键：(x,y,z)元组，值：标签
            with open(pre_path, 'r') as f:
                lines = f.readlines()
                # 跳过PLY头部（直到end_header）
                start_idx = None
                for i, line in enumerate(lines):
                    if line.strip() == 'end_header':
                        start_idx = i + 1
                        break
                if start_idx is None:
                    print(f"文件{pre_path}无end_header，跳过")
                    continue

                # 读取数据行（x,y,z,label,...）
                for line in lines[start_idx:]:
                    parts = line.strip().split()
                    if len(parts) < 4:  # 至少需要x,y,z,label
                        continue
                    x, y, z = map(float, parts[:3])
                    label = int(parts[3])
                    # 用元组存储坐标（保留6位小数避免浮点数精度问题）
                    coord_key = (round(x, 6), round(y, 6), round(z, 6))
                    pre_coords[coord_key] = label
            
            # 2. 读取final_seg文件的坐标和标签，对比参考标签
            total_matched = 0  # 当前文件坐标匹配的点总数
            correct_matched = 0  # 当前文件标签一致的匹配点数量
            
            with open(final_path, 'r') as f:
                lines = f.readlines()
                start_idx = None
                for i, line in enumerate(lines):
                    if line.strip() == 'end_header':
                        start_idx = i + 1
                        break
                if start_idx is None:
                    print(f"文件{final_path}无end_header，跳过")
                    continue
                
                for line in lines[start_idx:]:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    x, y, z = map(float, parts[:3])
                    final_label = int(parts[3])
                    coord_key = (round(x, 6), round(y, 6), round(z, 6))
                    
                    # 检查坐标是否在pre_antenna的点中
                    if coord_key in pre_coords:
                        total_matched += 1
                        if final_label == pre_coords[coord_key]:
                            correct_matched += 1
            
            # 累加至总体统计
            total_matched_all += total_matched
            correct_matched_all += correct_matched
            
            # 3. 计算并输出当前文件准确率
            if total_matched == 0:
                print(f"文件 {stem}：无坐标匹配的点")
            else:
                accuracy = (correct_matched / total_matched) * 100
                print(f"文件 {stem}：")
                print(f"  坐标匹配的点总数：{total_matched}")
                print(f"  标签正确的点数量：{correct_matched}")
                print(f"  标签准确率：{accuracy:.2f}%\n")

        except Exception as e:
            print(f"处理文件 {stem} 时出错：{str(e)}\n")

    # 计算并输出总体准确率
    if total_matched_all == 0:
        print("所有文件中均无坐标匹配的点，无法计算总体准确率")
    else:
        overall_accuracy = (correct_matched_all / total_matched_all) * 100
        print("="*50)
        print(f"总体统计：")
        print(f"  所有文件的总匹配点数：{total_matched_all}")
        print(f"  所有文件的总正确匹配点数：{correct_matched_all}")
        print(f"  总体标签准确率：{overall_accuracy:.2f}%")

if __name__ == "__main__":
    # 文件夹路径
    pre_antenna_dir = "data/nasa/test/point2048"
    final_seg_dir = "final_seg"
    
    # 计算准确率
    calculate_label_accuracy(pre_antenna_dir, final_seg_dir)

