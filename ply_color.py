def add_color_to_ply(input_file, output_file):
    """
    为PLY文件中的点添加颜色信息
    
    参数:
        input_file: 输入PLY文件路径
        output_file: 输出PLY文件路径
    """
    try:
        # 读取原始PLY文件
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # 解析头部信息
        header_lines = []
        vertex_count = 0
        in_header = True
        
        for i, line in enumerate(lines):
            line = line.strip()
            header_lines.append(line)
            
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line == 'end_header':
                in_header = False
                break
        
        # 验证头部信息
        if vertex_count == 0:
            print("错误: 未找到顶点数量信息")
            return
        
        # 提取顶点数据
        vertex_lines = lines[len(header_lines):len(header_lines) + vertex_count]
        
        # 定义标签对应的颜色
        label_colors = {
            1: (255, 0, 0),    # 红色
            2: (0, 0, 255),    # 蓝色
            3: (255, 255, 0)   # 黄色
        }
        
        # 处理每个顶点，添加颜色信息
        colored_vertices = []
        for line in vertex_lines:
            parts = line.strip().split()
            if len(parts) < 4:  # 确保有x,y,z,label
                continue
                
            x, y, z = parts[0], parts[1], parts[2]
            label = int(float(parts[3]))  # 转换为整数标签
            
            # 获取标签对应的颜色，默认使用白色
            r, g, b = label_colors.get(label, (255, 255, 255))
            
            # 创建新的顶点行，包含颜色信息
            colored_vertex = f"{x} {y} {z} {label} {r} {g} {b}"
            colored_vertices.append(colored_vertex)
        
        # 创建新的头部，添加颜色属性定义
        new_header = []
        for line in header_lines:
            if line == 'end_header':
                # 在end_header之前插入颜色属性定义
                new_header.append('property uchar red')
                new_header.append('property uchar green')
                new_header.append('property uchar blue')
            new_header.append(line)
        
        # 写入新的PLY文件
        with open(output_file, 'w') as f:
            f.write('\n'.join(new_header) + '\n')
            f.write('\n'.join(colored_vertices) + '\n')
        
        print(f"成功处理文件: {input_file}")
        print(f"已保存带颜色的新文件: {output_file}")
        
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")

if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_file = "aura_2b.0.105000.two_component.Annotation.ply"
    output_file = "aura_2b.0.105000.two_component.Annotation_colored.ply"
    
    # 处理文件
    add_color_to_ply(input_file, output_file)