import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 全局配置
everything_mask_dir = "everything_masks"  # 保存所有掩码的目录

def show_anns(anns):
    """可视化分割结果"""
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def process_image(image_path, output_dir, mask_generator):
    """处理单张图片并保存分割结果"""
    # 加载图片
    image = Image.open(image_path)
    image_np = np.array(image)
    
    # 生成掩码
    masks = mask_generator.generate(image_np)
    
    # 创建输出子目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存原始图片
    original_path = os.path.join(output_dir, "original.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.axis('off')
    plt.savefig(original_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 保存分割结果图
    result_path = os.path.join(output_dir, "segmentation_result.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(result_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 创建掩码目录
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    
    # 保存所有掩码
    for i, mask in enumerate(masks):
        mask_path = os.path.join(mask_dir, f"mask_{i+1}.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(mask['segmentation'])
        plt.axis('off')
        plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    # 保存所有掩码的数组到everything_mask_dir
    all_masks = [mask['segmentation'] for mask in masks]
    # 构建在everything_mask_dir中的路径
    model_name = os.path.basename(os.path.dirname(image_path))
    view_name = os.path.splitext(os.path.basename(image_path))[0]
    all_masks_path = os.path.join(everything_mask_dir, model_name, f"{view_name}_all_masks.npy")
    os.makedirs(os.path.dirname(all_masks_path), exist_ok=True)
    np.save(all_masks_path, np.array(all_masks, dtype=object))  # 注意：使用dtype=object保存不同尺寸的掩码
    
    print(f"处理完成: {image_path} -> 生成 {len(masks)} 个分割区域")
    return masks

def batch_process_models(sam_checkpoint, input_base_dir="2d_model", output_base_dir="seg_everything", model_type="vit_h"):
    """批量处理所有模型和视角图片"""
    # 确保全局输出目录存在
    os.makedirs(everything_mask_dir, exist_ok=True)
    
    # 加载SAM模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # 创建掩码生成器
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    # 遍历所有模型目录
    model_dirs = [d for d in os.listdir(input_base_dir) 
                 if os.path.isdir(os.path.join(input_base_dir, d))]
    
    if not model_dirs:
        print(f"在 {input_base_dir} 中没有找到模型目录")
        return
    
    print(f"找到 {len(model_dirs)} 个模型目录，开始处理...")
    
    total_images = 0
    # 处理每个模型
    for model_name in model_dirs:
        model_dir = os.path.join(input_base_dir, model_name)
        
        # 查找所有图片文件
        image_files = [f for f in os.listdir(model_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not image_files:
            print(f"{model_name} ")
            continue
            
        print(f"{model_name} ({len(image_files)} )")
        
        # 处理每张视角图片
        for img_file in image_files:
            # 获取不带扩展名的文件名
            view_name = os.path.splitext(img_file)[0]
            
            # 输入图片路径
            image_path = os.path.join(model_dir, img_file)
            
            # 输出目录结构: output_base_dir/model_name/view_name/
            output_dir = os.path.join(output_base_dir, model_name, view_name)
            
            # 处理图片
            process_image(image_path, output_dir, mask_generator)
            total_images += 1
    
    print(f"\n处理完成! 共处理 {len(model_dirs)} 个模型，{total_images} 张图片")
    print(f"结果保存在: {output_base_dir}")
    print(f"所有掩码数组保存在: {everything_mask_dir}")

if __name__ == "__main__":
    # SAM模型路径
    sam_checkpoint = "ckpt/sam_vit_h_4b8939.pth"
    
    # 批量处理所有模型
    batch_process_models(sam_checkpoint)