## ZSMMPS 项目说明

本仓库实现了基于多视角投影与 SAM 的卫星部件分割流程，包括：
- 3D 点云特征提取与零样本分割（CLIP 投影 + 几何特征）
- 2D 多视角基于提示点与 Everything 的分割（SAM）
- 2D 掩码融合为部件类别（机身/机翼）
- 分割结果导出为 PLY、准确率/IoU 评估

### 环境依赖
- Python 3.8+
- PyTorch（CUDA 环境可选）
- 其他：opencv-python、matplotlib、Pillow、tqdm、numpy、plyfile、scikit-learn（可选，用于提示点聚类）、open3d（libs/lib_o3d 依赖）

示例安装（按需调整 CUDA/版本）：
```bash
pip install torch torchvision torchaudio
pip install opencv-python matplotlib pillow tqdm numpy plyfile scikit-learn open3d
pip install git+https://github.com/facebookresearch/segment-anything
```

SAM 权重：将 `sam_vit_h_4b8939.pth` 放置到 `ckpt/` 目录（可在 Meta 官方仓库获取）。

### 目录结构与数据格式

运行前请准备如下目录（必要时自行创建）：

- 3D 数据（固定读取路径）：
  - `redata/nasa/test/point2048/*.ply`
  - PLY 顶点需至少包含字段：`x y z label`
  - `label` 为部件类别ID（本项目约定 1/2/3，详见下文导出与评估）

- 2D 渲染图与提示点：
  - 图像：`2d_model/<模型名>/<视图名>.png`（脚本内部会统一为 800x600 尺寸处理）
  - 提示点：`point/<模型名>/<视图名>.txt`
    - 每行格式：`x y tag`
    - 用于机身分割（body.py）：`tag=1` 为正点，`tag=3` 为负点
    - 用于机翼分割（wing.py）：`tag=3` 为正点，`tag=1` 为负点

- SAM 权重：
  - `ckpt/sam_vit_h_4b8939.pth`

- 关键输出（脚本运行后自动产生）：
  - `seg_everything/` 与 `everything_masks/`：Everything 分割结果与掩码数组
  - `body_masks/`, `seg_body/`, `colored_bodyseg/`：机身掩码与可视化
  - `wing_masks/`, `seg_wing/`, `colored_wingseg/`：机翼掩码与可视化
  - `fused_body_wing_fixpoint/`：融合后的彩色语义掩码（红：机身，黄：机翼，黑：背景）
  - `output/<CLIP名>/satellite/<样本名>/`：3D 流水线中间结果（见下）
  - `prim_seg/`, `remove_seg/`, `prompt_ply/`：可选步骤产物（提示点/PLY 导出）

`output/<CLIP名>/satellite/<样本名>/` 典型文件：
- `pc.pt`：点云坐标 (N,3)
- `normal.pt`：法向量特征
- `fpfh.pt`：FPFH 几何特征
- `features.pt`：投影后的图像特征图
- `labels.pt`：类别ID
- `ifseen.pt`：可见性标记
- `pointloc.pt`：点在 2D 视域中的像素坐标
- `names.txt`：样本名称列表（位于 `output/<CLIP名>/satellite/` 目录）

### 一键上手（推荐流程）
1) 3D 特征提取与零样本搜索
```bash
# 重要：数据集路径在代码中固定为 redata/nasa
python part_run.py \
  --modelname "ViT-B/16" \
  --classchoice satellite \
  --device cuda:0 \
  --onlyevaluate True

# 脚本将遍历 redata/nasa/test/point2048 下的 PLY，
# 输出到 output/ViT-B_16/satellite/<样本名>/
```

2) 导出分割结果为 PLY（带颜色与标签）
```bash
python output_ply.py   # 推荐：逐样本读取 output/.../<样本名>/segpred.pt 与 pc.pt 并写出 PLY
# 或
python pt_plycolor.py  # 另一种导出方式，依赖聚合的 seg/pc
```


3) Everything 分割（2D，SAM）
```bash
python segeverthing.py  # 读取 2d_model/，输出 everything_masks/ 与 seg_everything/
```

4) 基于提示点的机身/机翼分割（2D，SAM）
```bash
python body.py  # 读取 2d_model/ 与 point/，输出 body_masks/、seg_body/、colored_bodyseg/
python wing.py  # 读取 2d_model/ 与 point/，输出 wing_masks/、seg_wing/、colored_wingseg/
```

5) 掩码融合（将 Everything 子区域映射为机身/机翼）
```bash
python fuse_b_w_mask.py  # 读取 everything_masks/、body_masks/、wing_masks/，输出 fused_body_wing_fixpoint/
```
python fuse_body_wing_antenna.py 融合wing
补充黑块


6) 准确率/IoU 评估
```bash
# 根据你真实/预测结果所在目录修改脚本尾部路径后运行
python 3d_anytenna_accuracy.py
```

### 可选：提示点自动聚类与导出（3D）

当你已经有带 label 的 PLY（例如来自 `remove_seg/`），可以从 3D 中聚类提取提示点并导出为 PLY：
```bash
python prompt_choose.py  # cluster_method 可选：kmeans / hdbscan / dbscan
# 输出到 prompt_ply/
```

### 常见问题
- 数据路径：`NASA_3D` 数据类当前在代码中固定为 `redata/nasa`，请将 PLY 放到 `redata/nasa/test/point2048/`。
- SAM 提示点标签：机身使用 `1` 为正点、`3` 为负点；机翼相反（`3` 为正点，`1` 为负点）。
- 结果导出：`output_ply.py` 依赖每个样本目录下的 `segpred.pt` 与 `pc.pt`。若未生成 `segpred.pt`，请确认 `partmodel/` 下的 `search_prompt` / `search_vweight` 过程已运行并保存预测。

### 主要脚本一览（按用途）
- 2D 分割：`segeverthing.py`, `body.py`, `wing.py`, `fuse.py`
- 3D 流水线：`part_run.py`, `data.py`, `libs/lib_o3d.py`（几何特征）
- 导出/可视化：`output_ply.py`, `pt_plycolor.py`
- 评估：`3d_anytenna_accuracy.py`
- 提示点（可选）：`prompt_choose.py`, `pt_plycolor.py`

如需扩展部件或数据类别，请从 `data.py` 的类别映射与 `body.py`/`wing.py` 的提示点规则入手修改。



