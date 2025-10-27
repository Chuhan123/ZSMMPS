import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement

def load_model_from_ply(ply_path):
    """从PLY文件加载点云和标签数据"""
    ply_data = PlyData.read(ply_path)
    vertex = ply_data['vertex']
    
    # 提取坐标和标签
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    labels = vertex['label']
    
    # 确保数据是连续的（修复内存对齐问题）
    points = np.ascontiguousarray(points)
    labels = np.ascontiguousarray(labels)
    
    # 随机采样2048个点（如果点数足够）
    if len(points) > 2048:
        indices = np.random.choice(len(points), 2048, replace=False)
        points = points[indices]
        labels = labels[indices]
    
    return (
        torch.tensor(points, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )

def kmean(model_pc, model_seg):
    """返回单个模型的采样点和对应部件标签"""
    # 处理空输入的情况
    if model_pc.numel() == 0 or model_seg.numel() == 0:
        return (
            torch.tensor([], device=model_pc.device),
            torch.tensor([], dtype=torch.long, device=model_pc.device)
        )
    
    unique_labels = torch.unique(model_seg)
    sampled_points = []
    sampled_labels = []

    for label in unique_labels:
        mask = model_seg == label
        class_points = model_pc[mask]
        n_points = class_points.size(0)

        # 小样本处理（所有标签通用）
        if n_points <= 5:
            if n_points == 0:  # 防止空的类别
                continue
            center = class_points.mean(dim=0)
            distances = torch.norm(class_points - center, dim=1)
            sampled_points.append(class_points[torch.argmin(distances)])
            sampled_labels.append(label)
            continue

        # 自适应参数计算
        eps = estimate_eps_tensor(class_points)
        min_samples = max(5, int(0.02 * n_points))

        # 执行DBSCAN聚类
        cluster_labels = dbscan_tensor(class_points, eps, min_samples)

        # 处理聚类结果
        unique_clusters = torch.unique(cluster_labels)
        valid_clusters = unique_clusters[unique_clusters != -1]

        # 通用聚类处理（标签2、3和其他标签）
        if len(valid_clusters) == 0:
            center = class_points.mean(dim=0)
            distances = torch.norm(class_points - center, dim=1)
            sampled_points.append(class_points[torch.argmin(distances)])
            sampled_labels.append(label)
        else:
            for cluster_id in valid_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_pts = class_points[cluster_mask]
                if len(cluster_pts) == 0:  # 防止空的聚类
                    continue
                center = cluster_pts.mean(dim=0)
                distances = torch.norm(cluster_pts - center, dim=1)
                sampled_points.append(cluster_pts[torch.argmin(distances)])
                sampled_labels.append(label)

    # 统一返回格式
    if sampled_points:
        return (
            torch.stack(sampled_points),
            torch.tensor(sampled_labels, dtype=torch.long, device=model_pc.device)
        )
    else:
        return (
            torch.tensor([], device=model_pc.device),
            torch.tensor([], dtype=torch.long, device=model_pc.device)
        )
    
    
def estimate_eps_tensor(points, k=5):
    """基于PyTorch的eps参数估计"""
    if len(points) <= k:
        return 0.1
    
    # 计算所有点之间的欧氏距离
    dist_matrix = torch.cdist(points, points)
    
    # 获取每个点的k近邻距离
    k_distances, _ = torch.topk(dist_matrix, k=k+1, largest=False)
    kth_distances = k_distances[:, k]
    
    # 使用中位数计算eps
    return 1.5 * torch.median(kth_distances)

def dbscan_tensor(points, eps, min_samples):
    """纯PyTorch实现的DBSCAN聚类"""
    # 处理空输入的情况
    if len(points) == 0:
        return torch.tensor([], dtype=torch.long, device=points.device)
    
    n = points.size(0)
    device = points.device
    
    # 计算距离矩阵
    dist_matrix = torch.cdist(points, points)
    
    # 标记核心点
    is_core = torch.sum(dist_matrix <= eps, dim=1) >= min_samples
    core_indices = torch.where(is_core)[0]
    
    # 初始化聚类标签
    labels = torch.full((n,), -1, dtype=torch.long, device=device)
    cluster_id = 0
    # 邻居查询缓存
    neighbors_cache = (dist_matrix <= eps).bool()
    for i in core_indices:
        if labels[i] != -1:
            continue
        # 广度优先搜索扩展聚类
        queue = [int(i)]
        labels[i] = cluster_id
        while queue:
            current = queue.pop(0)
            neighbors = torch.where(neighbors_cache[current])[0]
            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                    if is_core[neighbor]:  # 只扩展核心点的邻居
                        queue.append(neighbor.item())
        cluster_id += 1
    return labels

def prompt_ply(all_prompt_pc, all_prompt_label, model_names, output_dir="prompt_test"):
    os.makedirs(output_dir, exist_ok=True)

    # 颜色映射 (标签1,2,3对应黄蓝红)
    color_map = {
        1: [255, 255, 0],    # 黄色
        2: [0, 0, 255],    # 蓝色
        3: [255, 0, 0]     # 红色
    }

    # 遍历所有模型
    for i, (pc, labels) in enumerate(zip(all_prompt_pc, all_prompt_label)):
        # 从模型名称列表中获取对应的模型名称
        model_name = model_names[i] if i < len(model_names) else f"prompt_{i:03d}"
        
        # 移除文件名中的非法字符（保留字母、数字和常见符号）
        valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        safe_model_name = ''.join(c for c in model_name if c in valid_chars)
        if not safe_model_name:  # 若处理后名称为空，使用默认命名
            safe_model_name = f"prompt_{i:03d}"
        
        # 处理空的提示点情况
        if pc.numel() == 0 or labels.numel() == 0:
            print(f"警告: 模型 {model_name} 没有提示点，将生成空的PLY文件")
            pc_np = np.zeros((0, 3), dtype=np.float32)
            labels_np = np.zeros((0,), dtype=np.int32)
        else:
            # 将当前模型的点和标签转换为numpy
            pc_np = pc.cpu().numpy()
            labels_np = labels.cpu().numpy()
        
        # 确保点云是二维数组
        if pc_np.ndim == 1:
            pc_np = pc_np.reshape(-1, 3)
        
        n_points = pc_np.shape[0]
        
        # 为每个点生成颜色
        colors = np.zeros((n_points, 3), dtype=np.uint8)
        for label_id, color in color_map.items():
            if n_points > 0:  # 防止对空数组进行索引
                mask = (labels_np == label_id)
                colors[mask] = color
        
        # 创建结构化数组
        vertices = np.zeros(n_points, dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('label', 'i4'),
            ('red', 'u1'),
            ('green', 'u1'),
            ('blue', 'u1')
        ])
        
        if n_points > 0:  # 防止对空数组进行赋值
            vertices['x'] = pc_np[:, 0]
            vertices['y'] = pc_np[:, 1]
            vertices['z'] = pc_np[:, 2]
            vertices['label'] = labels_np
            vertices['red'] = colors[:, 0]
            vertices['green'] = colors[:, 1]
            vertices['blue'] = colors[:, 2]
        
        vertex_element = PlyElement.describe(vertices, 'vertex')
        ply_filename = os.path.join(output_dir, f"{safe_model_name}.ply")
        ply_data = PlyData([vertex_element], text=True)
        ply_data.write(ply_filename)
    
    print(f"保存完成! 共生成 {len(all_prompt_pc)} 个ASCII格式PLY文件到 {output_dir} 目录")

if __name__ == '__main__':
    # 设置PLY文件所在目录
    ply_dir = "remove_seg"  # 修改为你的PLY文件目录
    output_dir = "prompt_ply"  # 输出目录
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有PLY文件
    ply_files = [f for f in os.listdir(ply_dir) if f.endswith('.ply')]
    
    if not ply_files:
        print(f"错误: 在 {ply_dir} 目录下未找到PLY文件")
    else:
        print(f"找到 {len(ply_files)} 个PLY文件")
        
        all_prompt_pc = []
        all_prompt_label = []
        model_names = []
        
        # 遍历所有PLY文件
        for i, ply_file in enumerate(ply_files):
            # 获取模型名称（不包含扩展名）
            model_name = os.path.splitext(ply_file)[0]
            model_names.append(model_name)
            ply_path = os.path.join(ply_dir, ply_file)
            
            print(f"处理模型 {i+1}/{len(ply_files)}: {model_name}")
            
            try:
                # 加载点云和标签
                model_pc, model_seg = load_model_from_ply(ply_path)
                
                # 检查是否有GPU可用
                if torch.cuda.is_available():
                    model_pc = model_pc.cuda()
                    model_seg = model_seg.cuda()
                
                # 执行聚类采样
                prompt_pc, prompt_label = kmean(model_pc, model_seg)
                
                all_prompt_pc.append(prompt_pc)
                all_prompt_label.append(prompt_label)
                
                print(f"  处理完成，提取了 {len(prompt_pc)} 个提示点")
            except Exception as e:
                print(f"  处理失败: {str(e)}")
                # 添加空结果，保持索引一致性
                all_prompt_pc.append(torch.tensor([], dtype=torch.float32))
                all_prompt_label.append(torch.tensor([], dtype=torch.long))
        
        # 保存结果
        prompt_ply(all_prompt_pc, all_prompt_label, model_names, output_dir)