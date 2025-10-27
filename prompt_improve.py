import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.cluster import KMeans, HDBSCAN  # 新增sklearn聚类算法

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

def kmean(model_pc, model_seg, cluster_method='kmeans'):
    """返回单个模型的采样点和对应部件标签，支持多种聚类方法"""
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
        
        # 对于点数较少的类别，直接取中心点
        if n_points < 10:
            center = class_points.mean(dim=0)
            sampled_points.append(center)
            sampled_labels.append(label)
            continue

        # 转换为numpy用于sklearn算法
        class_points_np = class_points.cpu().numpy()
        
        # 根据选择的聚类方法执行聚类
        if cluster_method == 'kmeans':
            # K-Means聚类：根据点数自适应确定聚类数量
            n_clusters = min(5, max(1, n_points // 50))  # 每50个点一个聚类，最多5个
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(class_points_np)
            valid_clusters = np.unique(cluster_labels)
            
        elif cluster_method == 'hdbscan':
            # HDBSCAN聚类（无需指定聚类数量）
            hdbscan = HDBSCAN(min_samples=max(3, int(0.01 * n_points)), 
                             min_cluster_size=max(5, int(0.02 * n_points)))
            cluster_labels = hdbscan.fit_predict(class_points_np)
            valid_clusters = np.unique(cluster_labels)
            valid_clusters = valid_clusters[valid_clusters != -1]  # 排除噪声点
            
        else:  # 默认使用原DBSCAN
            eps = estimate_eps_tensor(class_points)
            min_samples = max(5, int(0.02 * n_points))
            cluster_labels = dbscan_tensor(class_points, eps, min_samples).cpu().numpy()
            valid_clusters = np.unique(cluster_labels)
            valid_clusters = valid_clusters[valid_clusters != -1]

        # 处理聚类结果
        if len(valid_clusters) == 0:
            # 无有效聚类时取中心点
            center = class_points.mean(dim=0)
            sampled_points.append(center)
            sampled_labels.append(label)
        else:
            for cluster_id in valid_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_pts = class_points[torch.tensor(cluster_mask, device=class_points.device)]
                if len(cluster_pts) == 0:
                    continue
                # 取聚类中心最近的点
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
    """基于PyTorch的eps参数估计（原DBSCAN使用）"""
    dist_matrix = torch.cdist(points, points)
    k_distances, _ = torch.topk(dist_matrix, k=k+1, largest=False)
    kth_distances = k_distances[:, k]
    return 1.5 * torch.median(kth_distances)

def dbscan_tensor(points, eps, min_samples):
    """纯PyTorch实现的DBSCAN聚类（保留原实现）"""
    if len(points) == 0:
        return torch.tensor([], dtype=torch.long, device=points.device)
    
    n = points.size(0)
    device = points.device
    
    dist_matrix = torch.cdist(points, points)
    is_core = torch.sum(dist_matrix <= eps, dim=1) >= min_samples
    core_indices = torch.where(is_core)[0]
    
    labels = torch.full((n,), -1, dtype=torch.long, device=device)
    cluster_id = 0
    neighbors_cache = (dist_matrix <= eps).bool()
    
    for i in core_indices:
        if labels[i] != -1:
            continue
        queue = [int(i)]
        labels[i] = cluster_id
        while queue:
            current = queue.pop(0)
            neighbors = torch.where(neighbors_cache[current])[0]
            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                    if is_core[neighbor]:
                        queue.append(neighbor.item())
        cluster_id += 1
    return labels

def prompt_ply(all_prompt_pc, all_prompt_label, model_names, output_dir="prompt_test"):
    # 保持原实现不变
    os.makedirs(output_dir, exist_ok=True)

    color_map = {
        1: [255, 255, 0],    # 黄色
        2: [0, 0, 255],    # 蓝色
        3: [255, 0, 0]     # 红色
    }

    for i, (pc, labels) in enumerate(zip(all_prompt_pc, all_prompt_label)):
        model_name = model_names[i] if i < len(model_names) else f"prompt_{i:03d}"
        
        valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        safe_model_name = ''.join(c for c in model_name if c in valid_chars)
        if not safe_model_name:
            safe_model_name = f"prompt_{i:03d}"
        
        if pc.numel() == 0 or labels.numel() == 0:
            print(f"警告: 模型 {model_name} 没有提示点，将生成空的PLY文件")
            pc_np = np.zeros((0, 3), dtype=np.float32)
            labels_np = np.zeros((0,), dtype=np.int32)
        else:
            pc_np = pc.cpu().numpy()
            labels_np = labels.cpu().numpy()
        
        if pc_np.ndim == 1:
            pc_np = pc_np.reshape(-1, 3)
        
        n_points = pc_np.shape[0]
        
        colors = np.zeros((n_points, 3), dtype=np.uint8)
        for label_id, color in color_map.items():
            if n_points > 0:
                mask = (labels_np == label_id)
                colors[mask] = color
        
        vertices = np.zeros(n_points, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('label', 'i4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        
        if n_points > 0:
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
    ply_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/ZSMMPS/remove_seg"
    output_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/ZSMMPS/prompt_ply"
    cluster_method = "dbscan"  # 'kmeans', 'hdbscan' 'dbscan'
    
    os.makedirs(output_dir, exist_ok=True)
    ply_files = [f for f in os.listdir(ply_dir) if f.endswith('.ply')]
    
    if not ply_files:
        print(f"错误: 在 {ply_dir} 目录下未找到PLY文件")
    else:
        print(f"找到 {len(ply_files)} 个PLY文件，使用{cluster_method}聚类方法")

        all_prompt_pc = []
        all_prompt_label = []
        model_names = []

        for i, ply_file in enumerate(ply_files):
            model_name = os.path.splitext(ply_file)[0]
            model_names.append(model_name)
            ply_path = os.path.join(ply_dir, ply_file)
            
            print(f"处理模型 {i+1}/{len(ply_files)}: {model_name}")
            
            try:
                model_pc, model_seg = load_model_from_ply(ply_path)
                
                if torch.cuda.is_available():
                    model_pc = model_pc.cuda()
                    model_seg = model_seg.cuda()
                
                # 使用指定的聚类方法
                prompt_pc, prompt_label = kmean(model_pc, model_seg, cluster_method)
                
                all_prompt_pc.append(prompt_pc)
                all_prompt_label.append(prompt_label)
                
                print(f"  处理完成，提取了 {len(prompt_pc)} 个提示点")
            except Exception as e:
                print(f"  处理失败: {str(e)}")
                all_prompt_pc.append(torch.tensor([], dtype=torch.float32))
                all_prompt_label.append(torch.tensor([], dtype=torch.long))

        prompt_ply(all_prompt_pc, all_prompt_label, model_names, output_dir)