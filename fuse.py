import os
import numpy as np
import cv2

everything_mask_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/ZSMMPS/everything_masks"
body_mask_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/ZSMMPS/body_masks"
wing_mask_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/ZSMMPS/wing_masks"
output_dir = "/media/yangxilab/DiskB/fch/GeoZe_improve/ZSMMPS/fused_body_wing_fixpoint"
os.makedirs(output_dir, exist_ok=True)

# 颜色定义 (RGB)
COLOR_BODY = (255, 0, 0)      # 红
COLOR_WING = (255, 255, 0)    # 黄
COLOR_BG = (0, 0, 0)          # 黑

for model_name in os.listdir(everything_mask_dir):
    model_every_dir = os.path.join(everything_mask_dir, model_name)
    if not os.path.isdir(model_every_dir):
        continue
    model_body_dir = os.path.join(body_mask_dir, model_name)
    model_wing_dir = os.path.join(wing_mask_dir, model_name)
    model_out_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    for fname in os.listdir(model_every_dir):
        view_name = fname.replace(".npy", "")
        everything_masks = np.load(os.path.join(model_every_dir, fname), allow_pickle=True)
        # 加载body和wing掩码
        body_mask_path = os.path.join(model_body_dir, f"{view_name}_wing_part_mask.png")
        wing_mask_path = os.path.join(model_wing_dir, f"{view_name}_wing_part_mask.png")
        if not (os.path.exists(body_mask_path) and os.path.exists(wing_mask_path)):
            print(f"缺少body或wing掩码: {body_mask_path}, {wing_mask_path}")
            continue
        body_mask = cv2.imread(body_mask_path, cv2.IMREAD_GRAYSCALE) > 0
        wing_mask = cv2.imread(wing_mask_path, cv2.IMREAD_GRAYSCALE) > 0

        h, w = body_mask.shape
        fused_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 计算每个掩码的大小并排序（从大到小）
        mask_sizes = []
        for i, fine_mask in enumerate(everything_masks):
            region = np.where(fine_mask)
            region_size = len(region[0])
            mask_sizes.append((i, region_size))
        
        # 按大小从大到小排序
        mask_sizes.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [i for i, _ in mask_sizes]
        
        # 按排序后的顺序处理掩码，小掩码会覆盖大掩码的重复区域
        for idx in sorted_indices:
            fine_mask = everything_masks[idx]
            region = np.where(fine_mask)
            region_size = len(region[0])
            if region_size > 0.2 * h * w:
                fused_mask[region] = COLOR_BG
                continue
            # 统计body/wing在该块内的True数
            body_count = np.sum(body_mask[region])
            wing_count = np.sum(wing_mask[region])
            
            # 根据统计结果设置颜色，这将覆盖之前较大掩码的设置
            if wing_count >= body_count:
                fused_mask[region] = COLOR_WING
            else:
                fused_mask[region] = COLOR_BODY

            # 检查最终结果中是否存在红色（身体部分）
            has_body = np.any(np.all(fused_mask == COLOR_BODY, axis=-1))

            # 如果没有身体部分，且body_mask中1的数量不超过0.1*600*800，才添加红色身体区域
            if not has_body:
                body_pixel_count = np.sum(body_mask)
                threshold = 0.1 * 600 * 800
                if body_pixel_count <= threshold:
                    body_coords = np.where(body_mask)
                    fused_mask[body_coords] = COLOR_BODY

        # 保存彩色mask
        out_path = os.path.join(model_out_dir, f"{view_name}.png")
        cv2.imwrite(out_path, cv2.cvtColor(fused_mask, cv2.COLOR_RGB2BGR))
        print(f"保存: {out_path}")