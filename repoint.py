import numpy as np
import sys

def process_ply(input_file, output_file):
    # 读取头部信息
    with open(input_file, 'r') as f:
        lines = f.readlines()
        header_end_idx = lines.index('end_header\n')
    
    header = lines[:header_end_idx + 1]
    data_lines = lines[header_end_idx + 1:]
    
    # 提取顶点和面的数量
    vertex_count = 0
    face_count = 0
    for line in header:
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[2])
        elif line.startswith('element face'):
            face_count = int(line.split()[2])
    
    # 读取顶点数据
    vertex_data = np.genfromtxt(data_lines[:vertex_count], dtype=str)
    
    # 删除包含nan的行
    valid_mask = ~np.isnan(vertex_data[:, :3].astype(float)).any(axis=1)
    vertex_data = vertex_data[valid_mask]
    
    # 将第4列(red)转换为整数
    vertex_data[:, 3] = np.round(vertex_data[:, 3].astype(float)).astype(int)
    
    # 更新头部中的顶点数量
    new_vertex_count = len(vertex_data)
    for i, line in enumerate(header):
        if line.startswith('element vertex'):
            header[i] = f'element vertex {new_vertex_count}\n'
            break
    
    # 写入处理后的文件
    with open(output_file, 'w') as f:
        f.writelines(header)
        np.savetxt(f, vertex_data, fmt='%s')
        f.writelines(data_lines[vertex_count:])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python process_ply.py input.ply output.ply")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    process_ply(input_file, output_file)
    print(f"处理完成，结果已保存到 {output_file}")