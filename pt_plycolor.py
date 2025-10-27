from plyfile import PlyData, PlyElement
import numpy as np
import torch
import os
import os.path as osp

def save_ply(pc, pred_seg, output_path_ply, ascii_format=True):
    """
    保存点云为Ply格式，包含label和颜色属性
    
    参数:
    pc: 点云坐标，torch.Tensor或numpy数组
    pred_seg: 分割标签，torch.Tensor或numpy数组
    output_path_ply: 保存路径
    ascii_format: 是否使用ASCII格式，默认为True
    """
    # 转换为numpy数组
    if isinstance(pred_seg, torch.Tensor):
        pred_seg = pred_seg.cpu().numpy()
    if isinstance(pc, torch.Tensor):
        pc = pc.cpu().numpy()
    
    # 调整形状
    pc = pc.reshape(-1, 3)
    pred_seg = pred_seg.reshape(-1).astype(int)
    
    # 定义颜色映射 (1: 黄色, 2: 蓝色, 3: 红色, 其他: 白色)
    color_map = {
        1: (255, 255, 0),    # 黄色
        2: (0, 0, 255),      # 蓝色
        3: (255, 0, 0),      # 红色
    }
    
    # 创建颜色数组，默认为白色
    colors = np.ones((len(pc), 3), dtype=np.uint8) * 255
    
    # 根据标签设置颜色
    for label, color in color_map.items():
        mask = pred_seg == label
        colors[mask] = color
    
    # 定义顶点数据类型（包含x,y,z,label和颜色属性）
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('label', 'i4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]
    vertices = np.empty(len(pc), dtype=vertex_dtype)
    
    # 填充顶点数据
    vertices['x'] = pc[:, 0]
    vertices['y'] = pc[:, 1]
    vertices['z'] = pc[:, 2]
    vertices['label'] = pred_seg
    vertices['red'] = colors[:, 0]
    vertices['green'] = colors[:, 1]
    vertices['blue'] = colors[:, 2]
    
    # 创建Ply元素
    ply_element = PlyElement.describe(vertices, 'vertex')
    
    # 保存为Ply文件
    PlyData([ply_element], text=ascii_format).write(output_path_ply)

def save_prompt(prompt_points, file_path):
    """
    将提示点保存为Ply文件
    
    参数:
    prompt_points: 提示点坐标，torch.Tensor或numpy数组
    file_path: 保存路径
    """
    if isinstance(prompt_points, torch.Tensor):
        prompt_points = prompt_points.cpu().numpy()
    vertex = np.zeros(prompt_points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex['x'] = prompt_points[:, 0]
    vertex['y'] = prompt_points[:, 1]
    vertex['z'] = prompt_points[:, 2]

    ply = PlyData([PlyElement.describe(vertex, 'vertex')])
    ply.write(file_path)

def save_prompt_points_to_ply(prompt_points, file_path):
    """
    将第一批次的提示点坐标保存为 .ply 文件

    参数：
    prompt_points: [N, C*max_clusters, 3] 提示点坐标
    file_path: 保存的 .ply 文件路径
    """
    # 只取第一批次的提示点
    first_batch_points = prompt_points[0].cpu().numpy()
    vertex = np.zeros(first_batch_points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex['x'] = first_batch_points[:, 0]
    vertex['y'] = first_batch_points[:, 1]
    vertex['z'] = first_batch_points[:, 2]
    ply = PlyData([PlyElement.describe(vertex, 'vertex')])
    ply.write(file_path)

def read_names_file(names_file_path):
    """
    读取名称文件，返回名称列表
    
    参数:
    names_file_path: 名称文件路径
    
    返回:
    名称列表
    """
    if not os.path.exists(names_file_path):
        print(f"警告: 名称文件 '{names_file_path}' 不存在，将使用默认名称")
        return []
    
    with open(names_file_path, 'r') as f:
        names = [line.strip() for line in f if line.strip()]
    return names

if __name__ == '__main__':
    # 设置输出路径
    output_path = 'output/ViT-B_16/satellite'
    os.makedirs(output_path, exist_ok=True)
    
    seg = torch.load(osp.join(output_path, "segpred.pt"))
    pc = torch.load(osp.join(output_path, "pc.pt"))
    label = torch.load(osp.join(output_path, "labels.pt"))
    # sam_label = torch.load(osp.join(output_path, "samseg_result.pt"))
    
    # 读取名称文件
    names_file = osp.join("names.txt")
    model_names = read_names_file(names_file)
    
    num_samples = len(pc)
    
    # 确定使用的名称列表
    if not model_names:
        # 如果没有提供名称文件或文件为空，使用默认名称
        model_names = [f"model_{i}" for i in range(num_samples)]
    else:
        # 确保名称数量与样本数量匹配
        if len(model_names) < num_samples:
            print(f"警告: 名称数量({len(model_names)})少于样本数量({num_samples})，将重复使用名称")
            model_names = [model_names[i % len(model_names)] for i in range(num_samples)]
        elif len(model_names) > num_samples:
            print(f"警告: 名称数量({len(model_names)})多于样本数量({num_samples})，将只使用前{num_samples}个名称")
            model_names = model_names[:num_samples]
    output_dir = 'prim_seg'
    # 保存每个样本的分割结果
    for i in range(num_samples):
        model_name = model_names[i]
        seg_ply = osp.join(output_dir, f'{model_name}.ply')
        save_ply(pc[i], seg[i], seg_ply)
        print(f"已保存分割结果: {seg_ply}")    