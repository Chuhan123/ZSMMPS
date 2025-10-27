import os
import numpy as np
from scipy.spatial import KDTree

def read_ply_with_label(filename):
    """读取包含label属性的PLY文件，返回点坐标、标签和头信息"""
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # 解析PLY头
    header_end = -1
    vertex_count = 0
    label_index = -1
    properties = []
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if line_lower.startswith('element vertex'):
            vertex_count = int(line_lower.split()[-1])
        elif line_lower.startswith('property'):
            # 记录所有属性及其顺序
            parts = line_lower.split()
            if len(parts) >= 3:
                prop_type, prop_name = parts[1], parts[2]
                properties.append(prop_name)
                if prop_name == 'label':
                    # 标签的索引是属性列表的长度-1
                    label_index = len(properties) - 1
        elif line_lower == 'end_header':
            header_end = i + 1
            break
    
    if header_end == -1 or vertex_count == 0 or label_index == -1:
        raise ValueError("PLY文件格式不符合预期，未找到必要的头信息")
    
    # 读取点数据
    points = []
    labels = []
    all_data = []  # 保存所有数据用于后续写入
    
    # 数据行的起始和结束索引
    data_start = header_end
    data_end = header_end + vertex_count
    
    for line in lines[data_start:data_end]:
        if not line:  # 跳过空行
            continue
            
        parts = line.split()
        # 提取x,y,z坐标（前三个属性）
        x, y, z = map(float, parts[:3])
        # 提取label（根据属性列表中的索引）
        label = int(parts[label_index])
        
        points.append([x, y, z])
        labels.append(label)
        all_data.append(parts)  # 保存原始数据用于写入
    
    return np.array(points), np.array(labels), all_data, lines[:header_end], label_index

def write_ply_with_label(filename, labels, all_data, header_lines, label_index):
    """写入包含label属性的PLY文件，保留所有原始属性，并在label为2时同步将颜色改为蓝色(0,0,255)"""
    # 解析header获取vertex属性顺序
    prop_names = []
    in_vertex = False
    for line in header_lines:
        low = line.lower().strip()
        if low.startswith('element vertex'):
            in_vertex = True
            continue
        if low.startswith('element ') and not low.startswith('element vertex'):
            in_vertex = False
        if in_vertex and low.startswith('property'):
            parts = low.split()
            if len(parts) >= 3:
                prop_names.append(parts[2])

    name_to_idx = {n: i for i, n in enumerate(prop_names)}
    red_idx = name_to_idx.get('red', -1)
    green_idx = name_to_idx.get('green', -1)
    blue_idx = name_to_idx.get('blue', -1)

    with open(filename, 'w') as f:
        # 写入头信息
        f.write('\n'.join(header_lines) + '\n')

        # 写入点数据
        for i in range(len(labels)):
            if i >= len(all_data):
                continue

            parts = all_data[i].copy()

            # 更新label
            if label_index != -1 and label_index < len(parts):
                lbl = int(labels[i])
                parts[label_index] = str(lbl)
                # 若该点label为2，强制颜色为蓝色
                if lbl == 2 and red_idx != -1 and green_idx != -1 and blue_idx != -1:
                    parts[red_idx] = '0'
                    parts[green_idx] = '0'
                    parts[blue_idx] = '255'

            f.write(' '.join(parts) + '\n')

def update_labels_from_pre_antenna(remove_seg_dir, pre_antenna_dir, output_dir=None):
    """
    对比pre_antenna和remove_seg目录中的同名文件，
    将pre_antenna中标签为2的点替换到remove_seg文件中对应点的标签
    
    参数:
    remove_seg_dir: 经过初步处理的点云文件目录
    pre_antenna_dir: 参考点云文件目录
    output_dir: 输出目录，默认为在remove_seg_dir同级创建updated_seg目录
    """
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(remove_seg_dir), 'updated_seg')
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取两个目录中的所有PLY文件
    remove_seg_files = [f for f in os.listdir(remove_seg_dir) if f.endswith('.ply')]
    pre_antenna_files = [f for f in os.listdir(pre_antenna_dir) if f.endswith('.ply')]
    
    # 找到同名文件
    common_files = list(set(remove_seg_files) & set(pre_antenna_files))
    
    if not common_files:
        print("没有找到同名的PLY文件进行对比处理")
        return
    
    # 处理每个同名文件
    for filename in common_files:
        remove_seg_path = os.path.join(remove_seg_dir, filename)
        pre_antenna_path = os.path.join(pre_antenna_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # 读取remove_seg文件
            rs_points, rs_labels, rs_all_data, rs_header, rs_label_idx = read_ply_with_label(remove_seg_path)
            
            # 读取pre_antenna文件
            pa_points, pa_labels, _, _, _ = read_ply_with_label(pre_antenna_path)
            
            # 找到pre_antenna中标签为2的点
            pa_label_2_indices = np.where(pa_labels == 2)[0]
            if len(pa_label_2_indices) == 0:
                print(f"文件 {filename} 的pre_antenna版本中没有标签为2的点，直接复制remove_seg文件")
                # 直接复制文件
                with open(remove_seg_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
                continue
            
            # 构建KDTree用于查找对应点
            kdtree = KDTree(rs_points)
            
            # 记录被替换的点数量
            replaced_count = 0
            
            # 对pre_antenna中每个标签为2的点，找到remove_seg中对应点并更新标签
            for idx in pa_label_2_indices:
                # 查找最近邻点（认为是对应点）
                dist, nearest_idx = kdtree.query(pa_points[idx])
                
                # 设置一个距离阈值，确保是同一个点（根据实际数据尺度调整）
                distance_threshold = 1e-6
                if dist < distance_threshold:
                    # 将remove_seg中的点标签更新为2
                    rs_labels[nearest_idx] = 2
                    replaced_count += 1
            
            # 写入处理后的文件
            write_ply_with_label(output_path, rs_labels, rs_all_data, rs_header, rs_label_idx)
            print(f"已处理 {filename}，从pre_antenna更新了 {replaced_count} 个标签为2的点")
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 设置目录路径
    remove_seg_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/ZSMMPS/remove_seg"
    pre_antenna_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/ZSMMPS/pre_antenna"
    output_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/ZSMMPS/final_seg"  # 新增输出目录变量
    
    # 执行更新操作，指定输出目录为final_antenna
    update_labels_from_pre_antenna(remove_seg_dir, pre_antenna_dir, output_dir)
    print("所有文件处理完成")
    