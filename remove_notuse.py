import os
import numpy as np
from pathlib import Path

def read_ply_file(file_path):
    """读取PLY文件并返回顶点数据和标签"""
    points = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 解析头部信息
    header = []
    vertex_count = 0
    property_names = []
    in_header = True
    
    for line in lines:
        line = line.strip()
        header.append(line)
        
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        elif line.startswith('property'):
            parts = line.split()
            if len(parts) >= 3:
                property_names.append(parts[2])
        elif line == 'end_header':
            in_header = False
            break
    
    # 验证必要属性
    required_properties = ['x', 'y', 'z', 'label']
    for prop in required_properties:
        if prop not in property_names:
            print(f"错误: 文件 {file_path} 缺少必要属性 {prop}")
            return [], header, 0
    
    # 获取各属性的索引
    x_idx = property_names.index('x')
    y_idx = property_names.index('y')
    z_idx = property_names.index('z')
    label_idx = property_names.index('label')
    
    # 检查是否有RGB属性
    has_rgb = 'red' in property_names and 'green' in property_names and 'blue' in property_names
    if has_rgb:
        r_idx = property_names.index('red')
        g_idx = property_names.index('green')
        b_idx = property_names.index('blue')
    
    # 解析顶点数据
    data_lines = lines[len(header):]
    actual_vertices = min(len(data_lines), vertex_count)
    
    for line in data_lines[:actual_vertices]:
        parts = line.strip().split()
        if len(parts) < max(label_idx, r_idx if has_rgb else 0) + 1:
            continue
            
        # 提取基本坐标和标签
        x = float(parts[x_idx])
        y = float(parts[y_idx])
        z = float(parts[z_idx])
        label = int(float(parts[label_idx]))  # 保持原转换方式，但添加日志
        
        # 提取RGB (如果有)
        if has_rgb:
            r = int(parts[r_idx])
            g = int(parts[g_idx])
            b = int(parts[b_idx])
            points.append((x, y, z, label, r, g, b))
        else:
            points.append((x, y, z, label))
    
    print(f"读取文件 {file_path}: 声明顶点数 {vertex_count}, 实际解析 {len(points)}")
    return points, header, len(points)

def write_ply_file(file_path, points, header_lines):
    """写入PLY文件，更新顶点数量"""
    with open(file_path, 'w') as f:
        # 写入header，更新顶点数量
        for line in header_lines:
            if line.startswith('element vertex'):
                f.write(f'element vertex {len(points)}\n')
            else:
                f.write(line + '\n')
    
        # 写入顶点数据
        for point in points:
            if len(point) > 4:  # 包含RGB
                f.write(f"{point[0]} {point[1]} {point[2]} {point[3]} {point[4]} {point[5]} {point[6]}\n")
            else:  # 不包含RGB
                f.write(f"{point[0]} {point[1]} {point[2]} {point[3]}\n")

def process_ply_files(pre_antenna_dir, test_dir, output_dir):
    """处理所有PLY文件，删除标签为2的点"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取pre_antenna目录下的所有PLY文件
    pre_antenna_files = list(Path(pre_antenna_dir).glob('*.ply'))
    
    for pre_file in pre_antenna_files:
        file_name = pre_file.name
        
        # 找到对应的测试文件
        test_file = Path(test_dir) / file_name
        if not test_file.exists():
            print(f"警告: 未找到对应的测试文件 {test_file}")
            continue
        
        try:
            # 读取原始文件
            original_points, header, orig_point_count = read_ply_file(pre_file)
            
            # 读取测试文件中的标签
            test_points, _, test_point_count = read_ply_file(test_file)
            labels = [point[3] for point in test_points]
            
            # 确保两个文件的点数相同
            if len(original_points) != len(test_points):
                print(f"警告: 文件 {file_name} 的原始点云和测试点云点数不一致: {len(original_points)} vs {len(test_points)}")
                # 取较小的点数
                min_count = min(len(original_points), len(test_points))
                original_points = original_points[:min_count]
                labels = labels[:min_count]
            
            # 统计标签为2的点
            label_2_count = sum(1 for label in labels if label == 2)
            
            # 删除标签为2的点
            filtered_points = [p for p, label in zip(original_points, labels) if label != 2]
            
            # 计算实际删除的点数
            points_removed = len(original_points) - len(filtered_points)
            
            # 写入处理后的文件
            output_file = Path(output_dir) / file_name
            write_ply_file(output_file, filtered_points, header)
            
            print(f"处理完成: {file_name}")
            print(f"  原始点数: {len(original_points)}, 处理后点数: {len(filtered_points)}")
            print(f"  标签为2的点: {label_2_count}, 实际删除: {points_removed}")
        
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")

if __name__ == "__main__":
    # 设置目录路径
    pre_antenna_dir = "data/nasa/test/point2048"
    test_dir = "pre_antenna"
    output_dir = "redata/nasa/test/point2048"
    
    # 验证输入目录是否存在
    if not Path(pre_antenna_dir).exists():
        print(f"错误: 目录 {pre_antenna_dir} 不存在")
        exit(1)
    
    if not Path(test_dir).exists():
        print(f"错误: 目录 {test_dir} 不存在")
        exit(1)
    
    # 处理文件
    process_ply_files(pre_antenna_dir, test_dir, output_dir)