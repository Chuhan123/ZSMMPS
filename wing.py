import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 配置参数
image_base_dir = "2d_model"  # 包含多个模型目录的基目录
prompt_base_dir = "point"        # 包含多个模型提示点目录的基目录
mask_output_dir = "wing_masks"  # 修改为point_masks
prompted_output_dir = "seg_wing"
colored_output_dir = "colored_wingseg"  # 新增彩色分割输出目录
model_type = "vit_h"  
checkpoint_path = "ckpt/sam_vit_h_4b8939.pth"

# 创建输出目录
os.makedirs(mask_output_dir, exist_ok=True)
os.makedirs(prompted_output_dir, exist_ok=True)
os.makedirs(colored_output_dir, exist_ok=True)  # 创建彩色分割输出目录

# 加载SAM模型
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamPredictor(sam)

# 定义颜色
positive_color = (0, 255, 0)   # 绿色作为正提示点 (标签3)
negative_color = (255, 0, 0)   # 红色作为负提示点 (标签1)
mask_color = (0, 0, 255)       # 蓝色作为掩码颜色

def show_mask(mask, ax, random_color=False):
    """显示掩码"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def process_model(model_name):
    """处理单个模型的所有视图（尺寸800×600，无文字）"""
    print(f"\n{'='*50}")
    print(f"开始处理模型: {model_name}")
    print(f"{'='*50}")
    
    # 获取该模型下所有视图名
    model_dir = os.path.join(image_base_dir, model_name)
    image_files = [f for f in os.listdir(model_dir) if f.lower().endswith('.png')]
    view_names = [os.path.splitext(f)[0] for f in image_files]
    
    # 创建模型特定的输出目录
    model_mask_dir = os.path.join(mask_output_dir, model_name)
    model_prompted_dir = os.path.join(prompted_output_dir, model_name)
    model_colored_dir = os.path.join(colored_output_dir, model_name)
    
    os.makedirs(model_mask_dir, exist_ok=True)
    os.makedirs(model_prompted_dir, exist_ok=True)
    os.makedirs(model_colored_dir, exist_ok=True)
    
    # 处理每个视图
    for view_name in view_names:
        print(f"\n处理视图: {model_name}/{view_name}.png")
        
        # 构建图像路径
        image_path = os.path.join(image_base_dir, model_name, f"{view_name}.png")
        if not os.path.exists(image_path):
            print(f"  ! 警告: 找不到视图 {view_name}.png 的图像文件")
            print(f"  ! 警告: 找不到视图 {view_name}.png 的图像文件")
            continue
        
        # 构建提示点路径
        prompt_path = os.path.join(prompt_base_dir, model_name, f"{view_name}.txt")
        
        if not os.path.exists(prompt_path):
            print(f"  ! 警告: 找不到提示点文件 {prompt_path}")
            continue
        
        # 读取图像并调整尺寸为800×600
        image = cv2.imread(image_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = cv2.resize(image, (800, 600))  # 调整为目标尺寸
        
        # 读取提示点并缩放坐标
        try:
            points = np.loadtxt(prompt_path, dtype=int)
            if len(points) == 0:
                continue
            
            # 关键修复：如果只有一个点，将其转换为二维数组
            if points.ndim == 1:
                points = points.reshape(1, -1)
            
            # 坐标缩放处理
            h_original, w_original = image.shape[:2]
            h_target, w_target = 600, 800
            scale_x, scale_y = w_target / w_original, h_target / h_original
            points[:, 0] = (points[:, 0] * scale_x).astype(int)
            points[:, 1] = (points[:, 1] * scale_y).astype(int)
        except Exception as e:
            print(f"  ! 错误: 读取提示点失败 {prompt_path}: {e}")
            continue
        
        # 分离坐标和标签
        coords = points[:, :2]
        original_labels = points[:, 2]
        
        # 保存各部分的掩码
        part_masks = {}
        part_scores = {}
        
        # 定义翅膀分割规则（明确设置：正标签3，负标签1）
        parts = [
            {'name': 'wing_part', 'positive': [3], 'negative': [1], 'color': (0, 255, 255)},  # 黄色-wing
        ]

        # 设置SAM预测器
        predictor.set_image(original_image)
        
        # 分割翅膀部分
        for part in parts:
            part_name = part['name']
            print(f"  分割部分: {part_name}")
            
            # 标签处理：明确将标签3作为正点(1)，标签1作为负点(0)
            labels = np.zeros_like(original_labels)
            labels[np.isin(original_labels, part['positive'])] = 1  # 正点标记为1
            labels[np.isin(original_labels, part['negative'])] = 0  # 负点标记为0
            
            # 过滤有效点（只保留标签3和1的点）
            valid_mask = np.isin(original_labels, part['positive'] + part['negative'])
            if not np.any(valid_mask):
                print(f"  ! 警告: 没有找到有效的提示点 (标签3或1)")
                continue
            
            part_coords = coords[valid_mask]
            part_labels = labels[valid_mask]
            
            # 确保坐标和标签是二维数组（即使只有一个点）
            if part_coords.ndim == 1:
                part_coords = part_coords.reshape(1, -1)
            if part_labels.ndim == 1:
                part_labels = part_labels.reshape(-1, 1)
            
            # SAM预测
            masks, scores, logits = predictor.predict(
                point_coords=part_coords,
                point_labels=part_labels.flatten(),  # 确保是一维数组
                multimask_output=True
            )
            
            # 选择最佳掩码
            highest_score_idx = np.argmax(scores)
            mask = masks[highest_score_idx]
            score = scores[highest_score_idx]
            
            # 保存结果
            part_masks[part_name] = mask
            part_scores[part_name] = score
            mask_filename = f"{view_name}_{part_name}_mask.png"
            cv2.imwrite(os.path.join(model_mask_dir, mask_filename), (mask * 255).astype(np.uint8))
            print(f"    √ 完成: {part_name} (得分: {score:.3f})")
        
        # 创建彩色分割图像
        colored_seg = np.zeros_like(original_image)
        if 'wing_part' in part_masks:
            colored_seg[part_masks['wing_part']] = parts[0]['color']  # 黄色掩码
        
        # 融合图像
        alpha = 0.7
        blended = cv2.addWeighted(original_image, 1 - alpha, colored_seg, alpha, 0)
        colored_path = os.path.join(model_colored_dir, f"{view_name}_colored_seg.png")
        cv2.imwrite(colored_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
        # 创建可视化图像（无文字，尺寸800×600）
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])  # 填充画布
        ax.imshow(original_image)
        
        # 显示掩码
        if 'wing_part' in part_masks:
            mask = part_masks['wing_part']
            color = np.array(parts[0]['color'][::-1]) / 255.0
            color = np.append(color, 0.6)
            h, w = mask.shape
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
        
        # 显示提示点：标签3为绿色正点，标签1为红色负点
        pos_points = coords[original_labels == 3]
        neg_points = coords[original_labels == 1]
        
        # 处理可能的单点情况
        if pos_points.ndim == 1:
            pos_points = pos_points.reshape(1, -1)
        if neg_points.ndim == 1:
            neg_points = neg_points.reshape(1, -1)
            
        # ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=150, edgecolor='white', linewidth=1.25)
        # ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=150, edgecolor='white', linewidth=1.25)
        
        # 移除文字和边框
        ax.set_title("")
        ax.axis('off')
        
        # 强制输出800×600像素
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img_array = np.asarray(canvas.buffer_rgba())
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        img_array = cv2.resize(img_array, (800, 600))
        vis_path = os.path.join(model_prompted_dir, f"{view_name}.png")
        cv2.imwrite(vis_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        plt.close(fig)
        
        # 保存语义掩码
        semantic_masks = {
            'wing_part': part_masks.get('wing_part', np.zeros_like(mask) if 'mask' in locals() else None),
        }
        np.save(os.path.join(model_mask_dir, f"{view_name}_semantic_masks.npy"), semantic_masks)
    
    print(f"\n模型 {model_name} 处理完成!")

# 主处理流程
def main():
    # 获取所有模型名称
    image_models = set(os.listdir(image_base_dir))
    prompt_models = set()
    if os.path.exists(prompt_base_dir):
        prompt_models = set(os.listdir(prompt_base_dir))
    valid_models = sorted(list(image_models & prompt_models))
    
    print(f"找到 {len(valid_models)} 个有效模型:")
    for i, model in enumerate(valid_models, 1):
        print(f"{i}. {model}")
    
    # 处理每个模型
    for model_name in valid_models:
        process_model(model_name)
    
    print("\n所有模型处理完成!")

if __name__ == "__main__":
    main()
