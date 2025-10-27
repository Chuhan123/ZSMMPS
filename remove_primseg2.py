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
    
    return np.array(points), np.array(labels), all_data, lines[:header_end]

def write_ply_with_label(filename, points, labels, all_data, header_lines):
    """写入包含label属性的PLY文件，保留所有原始属性，并根据label同步更新颜色(red, green, blue)"""
    # 解析header，定位vertex元素内的属性顺序与索引
    prop_names = []
    in_vertex = False
    for line in header_lines:
        low = line.lower().strip()
        if low.startswith('element vertex'):
            in_vertex = True
            continue
        if low.startswith('element ') and not low.startswith('element vertex'):
            # 下一个element开始，结束收集
            in_vertex = False
        if in_vertex and low.startswith('property'):
            parts = low.split()
            if len(parts) >= 3:
                prop_names.append(parts[2])

    # 构建索引映射
    name_to_idx = {n: i for i, n in enumerate(prop_names)}
    label_idx = name_to_idx.get('label', -1)
    red_idx = name_to_idx.get('red', -1)
    green_idx = name_to_idx.get('green', -1)
    blue_idx = name_to_idx.get('blue', -1)

    # 颜色映射：1=黄, 2=蓝, 3=红
    color_map = {
        1: (255, 255, 0),
        2: (0, 0, 255),
        3: (255, 0, 0)
    }

    with open(filename, 'w') as f:
        # 写入头信息
        f.write('\n'.join(header_lines) + '\n')

        # 写入点数据（同步更新label与颜色）
        for i in range(len(points)):
            if i >= len(all_data):
                continue

            parts = all_data[i].copy()

            # 更新label
            if label_idx != -1 and label_idx < len(parts):
                lbl = int(labels[i])
                parts[label_idx] = str(lbl)
                # 同步更新颜色（如果存在red/green/blue属性）
                if red_idx != -1 and green_idx != -1 and blue_idx != -1:
                    if lbl in color_map:
                        r, g, b = color_map[lbl]
                        parts[red_idx] = str(int(r))
                        parts[green_idx] = str(int(g))
                        parts[blue_idx] = str(int(b))
            
            f.write(' '.join(parts) + '\n')

def replace_label_2(prim_seg_dir, output_dir):
    """
    将prim_seg目录中所有PLY点云文件里标签为2的点替换为最近的非2标签
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历目录中的所有文件
    for filename in os.listdir(prim_seg_dir):
        if filename.endswith('.ply'):  # 只处理PLY文件
            input_path = os.path.join(prim_seg_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # 直接读取PLY文件
                points, labels, all_data, header = read_ply_with_label(input_path)
                
                # 找到所有标签为2的点
                label_2_indices = np.where(labels == 2)[0]
                if len(label_2_indices) == 0:
                    print(f"文件 {filename} 中没有标签为2的点，直接复制")
                    # 直接复制文件
                    with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                    continue
                
                # 找到所有标签不为2的点
                non_label_2_indices = np.where(labels != 2)[0]
                if len(non_label_2_indices) == 0:
                    print(f"文件 {filename} 中所有点都是标签2，无法替换，跳过处理")
                    continue
                
                # 构建KDTree用于最近邻搜索
                kdtree = KDTree(points[non_label_2_indices])
                
                # 对每个标签为2的点，找到最近的非2标签点并替换其标签
                for idx in label_2_indices:
                    # 查找最近邻
                    dist, nearest_idx = kdtree.query(points[idx])
                    # 替换标签
                    labels[idx] = labels[non_label_2_indices[nearest_idx]]
                
                # 写入处理后的文件
                write_ply_with_label(output_path, points, labels, all_data, header)
                print(f"已处理 {filename}，替换了 {len(label_2_indices)} 个标签为2的点")
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")


if __name__ == "__main__":
    # 设置输入和输出目录
    prim_seg_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/ZSMMPS/prim_seg"
    output_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/ZSMMPS/remove_seg"
    
    # 执行替换操作
    replace_label_2(prim_seg_dir, output_dir)
    print("所有文件处理完成")
    