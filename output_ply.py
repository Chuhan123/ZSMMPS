# -*- coding: utf-8 -*-
from plyfile import PlyData, PlyElement
import numpy as np
import torch
import os.path as osp
import os
"将初始分割pt文件转化为Ply文件"
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
    
    print("Point cloud shape: {}".format(pc.shape))
    print("Label shape: {}".format(pred_seg.shape))
    print("Unique labels: {}".format(np.unique(pred_seg)))
    
    # 定义颜色映射 (1: 黄色, 2: 蓝色, 3: 红色)
    color_map = {
        1: (255, 255, 0),    # 黄色
        2: (0, 0, 255),      # 蓝色
        3: (255, 0, 0)       # 红色
    }
    
    # 创建颜色数组，默认为白色
    colors = np.ones((len(pc), 3), dtype=np.uint8) * 255
    
    # 根据标签设置颜色
    for label, color in color_map.items():
        # 确保mask是一维的布尔数组，并且长度与pc相同
        mask = np.zeros(len(pc), dtype=bool)
        mask[pred_seg == label] = True
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

if __name__ == '__main__':
    output_path = 'output/ViT-B_16/satellite'
    
    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))]
    for subdir in subdirs:
        print("\n处理文件夹: {}".format(subdir))
        current_path = os.path.join(output_path, subdir)
        # 加载数据
        seg = torch.load(os.path.join(current_path, "segpred.pt"))
        pc = torch.load(os.path.join(current_path, "pc.pt"))
        label = torch.load(os.path.join(current_path, "labels.pt"))
        # 创建输出目录
        output_dir = os.path.join("Loral_seg")
        os.makedirs(output_dir, exist_ok=True)
            
        pc_ply = os.path.join(output_dir, 'pc_{}.ply'.format(subdir))
        seg_ply = os.path.join(output_dir, '{}.ply'.format(subdir))
        save_ply(pc, seg, seg_ply)
        #save_ply(pc, label, pc_ply)
            
        print("完成文件夹 {} 的处理".format(subdir))
        