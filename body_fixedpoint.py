import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 配置参数
image_base_dir = "2d_model"  # 包含多个模型目录的基目录
mask_output_dir = "body_masks_fixpoint"
prompted_output_dir = "seg_body_fixpoint"
colored_output_dir = "colored_bodyseg"
model_type = "vit_h"  
checkpoint_path = "ckpt/sam_vit_h_4b8939.pth"

# 创建输出目录
os.makedirs(mask_output_dir, exist_ok=True)
os.makedirs(prompted_output_dir, exist_ok=True)
os.makedirs(colored_output_dir, exist_ok=True)

# 加载SAM模型
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device=device)
predictor = SamPredictor(sam)


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
            continue
        
        # 读取图像并调整尺寸为800×600（W×H）
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ! 警告: 无法读取图像 {image_path}")
            continue
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转RGB
        original_image = cv2.resize(original_image, (800, 600))  # 强制缩放为800×600
        
        # 【核心修改1：直接定义固定正提示点】
        # 格式：[x坐标, y坐标]，对应图像的宽和高（800×600范围内）
        fixed_pos_coords = np.array([
            [385, 300],
            [400, 285],
            [400, 315],
            [415, 300]
        ])
        # 标签：1表示正提示点（所有固定点都是正点）
        fixed_labels = np.ones(len(fixed_pos_coords), dtype=int)
        
        # 保存各部分的掩码
        part_masks = {}
        part_scores = {}
        
        # 定义分割规则（仅使用固定正提示点，无负提示点）
        parts = [
            {'name': 'wing_part', 'positive': [1], 'negative': [], 'color': (0, 0, 255)},  # 负提示点为空
        ]

        # 设置SAM预测器
        predictor.set_image(original_image)
        
        # 分割翅膀部分
        for part in parts:
            part_name = part['name']
            print(f"  分割部分body: {part_name}")
            
            # 【核心修改2：直接使用固定提示点，无需过滤】
            part_coords = fixed_pos_coords
            part_labels = fixed_labels
            
            # 确保有有效提示点
            if len(part_coords) == 0:
                print(f"  ! 警告: 无有效提示点用于分割 {part_name}")
                continue
            
            # SAM预测（多掩码输出，选分数最高的）
            masks, scores, logits = predictor.predict(
                point_coords=part_coords,
                point_labels=part_labels,
                multimask_output=True
            )
            
            # 选择最佳掩码
            highest_score_idx = np.argmax(scores)
            mask = masks[highest_score_idx]
            score = scores[highest_score_idx]
            print(f"  - 最佳掩码分数: {score:.4f}")
            
            # 保存结果
            part_masks[part_name] = mask
            part_scores[part_name] = score
            mask_filename = f"{view_name}_{part_name}_mask.png"
            cv2.imwrite(os.path.join(model_mask_dir, mask_filename), (mask * 255).astype(np.uint8))
        
        # 创建彩色分割图像
        colored_seg = np.zeros_like(original_image)
        if 'wing_part' in part_masks:
            colored_seg[part_masks['wing_part']] = parts[0]['color']  # 蓝色掩码
        
        # 融合图像（原图与彩色掩码叠加）
        alpha = 0.7
        blended = cv2.addWeighted(original_image, 1 - alpha, colored_seg, alpha, 0)
        colored_path = os.path.join(model_colored_dir, f"{view_name}_colored_seg.png")
        cv2.imwrite(colored_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
        # 创建可视化图像（含提示点和掩码，800×600无文字）
        fig = plt.figure(figsize=(8, 6), dpi=100)  # 8×6英寸，100dpi → 800×600像素
        ax = fig.add_axes([0, 0, 1, 1])  # 填充整个画布，无留白
        ax.imshow(original_image)
        
        # 显示掩码
        if 'wing_part' in part_masks:
            mask = part_masks['wing_part']
            color = np.array(parts[0]['color'][::-1]) / 255.0  # BGR转RGB并归一化
            color = np.append(color, 0.6)  # 增加透明度
            h, w = mask.shape
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
        
        
        # 移除文字和边框
        ax.set_title("")
        ax.axis('off')
        
        # 保存可视化图像
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img_array = np.asarray(canvas.buffer_rgba())
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        img_array = cv2.resize(img_array, (800, 600))  # 强制确认尺寸
        vis_path = os.path.join(model_prompted_dir, f"{view_name}.png")
        cv2.imwrite(vis_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        plt.close(fig)
        
        # 保存语义掩码（numpy格式）
        if 'wing_part' in part_masks:
            semantic_masks = {'wing_part': part_masks['wing_part']}
            np.save(os.path.join(model_mask_dir, f"{view_name}_semantic_masks.npy"), semantic_masks)
        else:
            print(f"  ! 警告: 未生成 {view_name} 的语义掩码")
    
    print(f"\n模型 {model_name} 处理完成!")

# 主处理流程
def main():
    # 获取所有模型名称（仅需图像目录存在即可，无需提示点目录）
    if not os.path.exists(image_base_dir):
        print(f"  ! 错误: 图像基目录 {image_base_dir} 不存在")
        return
    
    valid_models = [d for d in os.listdir(image_base_dir) if os.path.isdir(os.path.join(image_base_dir, d))]
    valid_models = sorted(valid_models)
    
    print(f"找到 {len(valid_models)} 个有效模型:")
    for i, model in enumerate(valid_models, 1):
        print(f"{i}. {model}")
    
    # 处理每个模型
    for model_name in valid_models:
        process_model(model_name)
    
    print("\n所有模型处理完成!")

if __name__ == "__main__":
    main()