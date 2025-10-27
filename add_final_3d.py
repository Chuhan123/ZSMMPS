import os
import shutil
from pathlib import Path

def process_files(pre_antenna_dir, prim_seg_dir, output_dir=None):
    """
    处理两个文件夹中的文件，将prim_seg中与pre_antenna中标签为2的点对应的点替换为标签2，
    并保持顶点数量不变。若pre_antenna文件中无标签为2的点，则直接复制prim_seg文件到输出目录
    
    参数:
        pre_antenna_dir: pre_antenna文件夹路径
        prim_seg_dir: prim_seg文件夹路径
        output_dir: 输出文件夹路径，默认为prim_seg_dir
    """
    # 确保输出文件夹存在
    if output_dir is None:
        output_dir = prim_seg_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取两个文件夹中的所有文件名
    pre_files = {Path(f).stem: f for f in os.listdir(pre_antenna_dir) 
                if os.path.isfile(os.path.join(pre_antenna_dir, f))}
    prim_files = {Path(f).stem: f for f in os.listdir(prim_seg_dir) 
                if os.path.isfile(os.path.join(prim_seg_dir, f))}
    
    # 找出名称匹配的文件
    matched_files = set(pre_files.keys()) & set(prim_files.keys())
    if not matched_files:
        print("警告：两个文件夹中没有找到名称匹配的文件！")
        return
    
    print(f"找到{len(matched_files)}对匹配的文件，开始处理...")
    
    # 处理每对匹配的文件
    for file_stem in matched_files:
        pre_file_path = os.path.join(pre_antenna_dir, pre_files[file_stem])
        prim_file_path = os.path.join(prim_seg_dir, prim_files[file_stem])
        output_file_path = os.path.join(output_dir, prim_files[file_stem])
        
        try:
            # 读取pre_antenna文件中标签为2的点，存储为集合便于查找
            # 使用元组(x,y,z)作为键，值为完整的点信息
            label2_points = set()
            with open(pre_file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:  # 至少需要x,y,z和label
                        try:
                            label = int(parts[3])
                            if label == 2:
                                x, y, z = map(float, parts[:3])
                                # 使用四舍五入处理浮点数精度问题
                                x_rounded = round(x, 6)
                                y_rounded = round(y, 6)
                                z_rounded = round(z, 6)
                                label2_points.add((x_rounded, y_rounded, z_rounded))
                        except ValueError:
                            continue  # 跳过格式错误的行
            
            if not label2_points:
                print(f"文件 {pre_file_path} 中没有找到标签为2的点，直接复制原始文件")
                # 直接复制prim_seg文件到输出目录
                shutil.copy2(prim_file_path, output_file_path)
                continue
            
            # 读取prim_seg文件内容并解析头部
            header = []
            data_lines = []
            is_header = True
            vertex_line_found = False
            original_vertex_count = 0
            
            with open(prim_file_path, 'r') as f:
                for line in f:
                    stripped_line = line.strip()
                    if is_header:
                        header.append(stripped_line)
                        # 查找element vertex行
                        if stripped_line.startswith('element vertex'):
                            vertex_line_found = True
                            try:
                                original_vertex_count = int(stripped_line.split()[-1])
                            except:
                                print(f"警告: 无法解析文件 {prim_file_path} 中的顶点数量")
                        # 头部结束标记
                        if stripped_line == 'end_header':
                            is_header = False
                            data_lines.append(line)  # end_header属于头部，但数据从下一行开始
                        else:
                            data_lines.append(line)  # 保留头部行的原始格式（包括换行符）
                    else:
                        data_lines.append(line)  # 数据部分
            
            # 处理数据行，替换匹配的点的标签
            updated_data_lines = []
            replaced_count = 0
            
            # 处理原始数据行（跳过头部）
            for line in data_lines[len(header):]:
                stripped_line = line.strip()
                parts = stripped_line.split()
                if len(parts) >= 3:  # 至少需要x,y,z
                    try:
                        x, y, z = map(float, parts[:3])
                        # 四舍五入处理浮点数精度问题
                        x_rounded = round(x, 6)
                        y_rounded = round(y, 6)
                        z_rounded = round(z, 6)
                        
                        # 检查是否在pre_antenna的标签2点集中
                        if (x_rounded, y_rounded, z_rounded) in label2_points:
                            # 替换标签为2，并设置颜色为蓝色(0,0,255)
                            if len(parts) >= 4:
                                # 保留原始x,y,z，替换标签和颜色
                                updated_line = f"{parts[0]} {parts[1]} {parts[2]} 2 0 0 255"
                            else:
                                # 如果原始行没有标签和颜色，添加它们
                                updated_line = f"{parts[0]} {parts[1]} {parts[2]} 2 0 0 255"
                            updated_data_lines.append(updated_line)
                            replaced_count += 1
                        else:
                            # 不匹配的点保持原样
                            updated_data_lines.append(stripped_line)
                    except ValueError:
                        # 格式错误的行保持原样
                        updated_data_lines.append(stripped_line)
                else:
                    # 不符合格式的行保持原样
                    updated_data_lines.append(stripped_line)
            
            # 写入处理后的文件
            with open(output_file_path, 'w') as f:
                # 写入头部（顶点数量不变）
                f.write('\n'.join(header) + '\n')
                # 写入更新后的数据部分
                f.write('\n'.join(updated_data_lines) + '\n')
            
            print(f"已处理文件: {file_stem}")
            print(f"  顶点总数: {original_vertex_count if vertex_line_found else '未知'}")
            print(f"  替换的点数: {replaced_count}")
            
        except Exception as e:
            print(f"处理文件 {file_stem} 时出错: {str(e)}")

if __name__ == "__main__":
    # 设置文件夹路径
    pre_antenna_dir = "pre_antenna"  
    prim_seg_dir = "remove_seg"       
    output_dir = "final_seg" 
    
    # 执行文件处理
    process_files(pre_antenna_dir, prim_seg_dir, output_dir)