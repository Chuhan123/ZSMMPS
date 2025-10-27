def count_label_2_points(ply_file):
    """
    统计PLY文件中标签为2的点的数量
    
    参数:
        ply_file: PLY文件路径
    
    返回:
        标签为2的点的数量
    """
    label_2_count = 0
    vertex_count = 0
    in_vertex_section = False
    
    with open(ply_file, 'r') as f:
        for line in f:
            line = line.strip()
                
                # 解析文件头
            if not in_vertex_section:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line == 'end_header':
                    in_vertex_section = True
                    print(f"找到 {vertex_count} 个顶点")
                continue
                
                # 解析顶点数据
            if in_vertex_section:
                parts = line.split() # 至少需要label字段
                label = int(parts[3])  # 假设label在第4列
                if label == 2:
                    label_2_count += 1

    return label_2_count

if __name__ == "__main__":
    # 替换为实际的PLY文件路径
    ply_file_path = "data/nasa/test/point2048/Loral-1300Com-main.ply"
    
    count = count_label_2_points(ply_file_path)
    print(f"文件中标签为2的点数量: {count}")