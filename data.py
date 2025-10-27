import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from plyfile import PlyData
import torch
from tqdm import tqdm

# 类别定义（根据NASA数据集实际情况）
id2cat = ['satellite']
cat2part = {'satellite': ['body', 'wing']}
id2part2cat = [['body', 'satellite'], ['wing', 'satellite']]

class NASA_3D(Dataset):
    def __init__(self, num_points=2048, partition='test', class_choice=None):
        # 强制指定类别（所有文件均属于satellite类别）
        self.id2cat = id2cat
        self.cat2id = {cat: idx for idx, cat in enumerate(id2cat)}

        # 固定数据路径
        data_path = 'redata/nasa'
        partition_path = os.path.join(data_path, partition, 'point2048')

        # 加载数据（现在返回文件名）
        self.data, self.label, self.seg, self.names = self.load_data_partseg(partition_path)

        # 处理空数据集
        if len(self.data) == 0:
            raise RuntimeError("Loaded empty dataset! Check data paths and file formats.")

        self.seg_num = [len(parts) for parts in cat2part.values()]  # [3]
        self.index_start = [1]  # 部件ID起始索引
        self.num_points = num_points
        self.partition = partition
        self.class_choice = class_choice

        # 处理类别筛选（本示例只有单个类别）
        if self.class_choice is not None:
            if self.class_choice not in self.cat2id:
                raise ValueError(f"Invalid class choice: {self.class_choice}")
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.names = [self.names[i] for i in indices.numpy()]  # 同步筛选文件名
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = sum(self.seg_num)
            self.seg_start_index = 1

    def __getitem__(self, item):
        pointcloud = self.data[item]  # 直接获取点云数据
        seg = self.seg[item]  # 直接获取分割标签
        name = self.names[item]  # 获取文件名
        return pointcloud.astype(np.float32), seg.astype(np.int64), name  # 返回三个值

    def __len__(self):
        return len(self.data)  # 直接返回列表长度

    @staticmethod
    def load_ply_file(file_path):
        """加载单个PLY文件（适配卫星数据集格式）"""
        try:
            plydata = PlyData.read(file_path)
            vertices = plydata['vertex']

            # 提取坐标和标签
            points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
            seg = vertices['label'].astype(np.int64)

            # 提取文件名（不带扩展名）
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # 所有文件都属于satellite类别
            return points, 'satellite', seg, file_name
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise

    def load_data_partseg(self, root_path):
        """加载指定分区的数据（现在返回文件名）"""
        print(f"Loading data from: {os.path.abspath(root_path)}")

        # 验证路径存在性
        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Data directory {root_path} does not exist!")

        ply_files = glob.glob(os.path.join(root_path, '*.ply'))
        if not ply_files:
            raise FileNotFoundError(f"No PLY files found in {root_path}")

        all_data = []
        all_labels = []
        all_segs = []
        all_names = []  # 存储文件名

        for ply_file in ply_files:
            points, class_name, seg, file_name = self.load_ply_file(ply_file)
            class_id = self.cat2id[class_name]

            all_data.append(points)  # 直接存储点云数据
            all_labels.append(np.array([class_id]))  # 存储类别ID
            all_segs.append(seg)  # 直接存储分割标签
            all_names.append(file_name)  # 存储文件名

        print(f"Successfully loaded {len(all_data)} samples")
        return all_data, all_labels, all_segs, all_names