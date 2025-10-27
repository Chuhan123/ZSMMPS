import os
import numpy as np
from plyfile import PlyData, PlyElement 
from scipy.spatial import cKDTree
"修正原始数据划分标签"
def read_ply_with_labels(file_path):
    """读取PLY文件并提取顶点和标签"""
    try:
        ply_data = PlyData.read(file_path)
        vertices = np.vstack([
            ply_data['vertex']['x'],
            ply_data['vertex']['y'],
            ply_data['vertex']['z']
        ]).T
        
        # 检查是否存在标签字段
        if 'label' in ply_data['vertex']:
            labels = ply_data['vertex']['label']
            # 确保标签为整数类型
            if not np.issubdtype(labels.dtype, np.integer):
                labels = labels.astype(np.int32)
        else:
            print(f"警告: 文件 {os.path.basename(file_path)} 不包含标签字段")
            labels = np.zeros(len(vertices), dtype=np.int32)
            
        return vertices, labels
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None, None

def compare_and_update_labels(data_vertices, data_labels, pre_vertices, pre_labels):
    """比较两个点云的标签并更新数据"""
    if data_vertices is None or pre_vertices is None:
        return data_labels
    
    # 提取两个文件中标签为2的点
    data_mask = data_labels == 2
    pre_mask = pre_labels == 2
    
    data_points_2 = data_vertices[data_mask]
    pre_points_2 = pre_vertices[pre_mask]
    
    # 创建一个布尔数组，用于标记data中哪些标签2的点在pre中也存在
    points_to_keep = np.zeros(len(data_points_2), dtype=bool)
    
    # 设置容差值，用于判断两个点是否相同
    tolerance = 1e-5
    
    # 比较点云（使用KDTree加速查找）
    if len(pre_points_2) > 0:
        tree = cKDTree(pre_points_2)
        distances, _ = tree.query(data_points_2, k=1)
        points_to_keep = distances < tolerance
    
    # 更新标签
    data_indices_2 = np.where(data_mask)[0]
    indices_to_change = data_indices_2[~points_to_keep]
    data_labels[indices_to_change] = 3
    
    print(f"共处理 {len(data_indices_2)} 个标签为2的点")
    print(f"其中 {np.sum(points_to_keep)} 个点在两个文件中都存在")
    print(f"{len(indices_to_change)} 个点仅存在于第一个文件中，已将其标签改为1")
    
    return data_labels

def write_ply_with_labels(file_path, vertices, labels):
    """以ASCII 1.0格式写入PLY文件"""
    try:
        # 确保标签为整数类型
        if not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype(np.int32)
        
        # 创建顶点数据结构
        vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('label', 'i4')  # 使用整数类型存储标签
        ]
        
        vertex_data = np.zeros(len(vertices), dtype=vertex_dtype)
        vertex_data['x'] = vertices[:, 0]
        vertex_data['y'] = vertices[:, 1]
        vertex_data['z'] = vertices[:, 2]
        vertex_data['label'] = labels
        
        # 创建PlyElement对象
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        
        # 创建PlyData对象并以ASCII格式写入
        ply_data = PlyData([vertex_element])
        
        # 写入文件（兼容旧版plyfile库）
        with open(file_path, 'w') as f:
            # 手动写入头部信息，指定格式为ascii 1.0
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            for comment in ply_data.comments:
                f.write(f'comment {comment}\n')
            for obj in ply_data.elements:
                f.write(f'element {obj.name} {len(obj.data)}\n')
                for prop in obj.properties:
                    f.write(f'property {prop.name}\n')
            f.write('end_header\n')
            
            # 写入数据
            for row in vertex_data:
                # 将每个顶点的数据转换为字符串并写入
                vertex_str = ' '.join(map(str, row))
                f.write(f'{vertex_str}\n')
        
        print(f"已保存ASCII 1.0格式文件到 {file_path}")
    except Exception as e:
        print(f"写入文件 {file_path} 时出错: {e}")

def main():
    # 文件路径
    data_file = "data/nasa/test/point2048/Loral-1300Com-main.ply"
    pre_file = "pre_antenna/Loral-1300Com-main.ply"
    
    # 输出文件路径
    output_dir = "output"
    output_file = os.path.join(output_dir, "Loral-1300Com-main_ascii.ply")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 读取两个文件的点云和标签
    print(f"正在读取文件: {data_file}")
    data_vertices, data_labels = read_ply_with_labels(data_file)
    
    print(f"正在读取文件: {pre_file}")
    pre_vertices, pre_labels = read_ply_with_labels(pre_file)
    
    if data_vertices is not None and pre_vertices is not None:
        # 比较并更新标签
        updated_labels = compare_and_update_labels(
            data_vertices, data_labels, pre_vertices, pre_labels
        )
        
        # 保存更新后的文件
        write_ply_with_labels(output_file, data_vertices, updated_labels)

if __name__ == "__main__":
    main()