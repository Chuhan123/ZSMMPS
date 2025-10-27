import os
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
# 在文件开头的导入区域添加
from concurrent.futures import as_completed

def process_view(fuse_path, body_mask_path, wing_mask_path, output_path):
    """处理单个视图图像，使用掩码进行像素补充"""
    try:
        # 确保输入文件存在
        if not os.path.exists(body_mask_path) or not os.path.exists(wing_mask_path) or not os.path.exists(fuse_path):
            print(f"跳过 {fuse_path}: 缺少对应的身体掩码、翅膀掩码或融合图像")
            return False

        # 加载融合图像和掩码
        fuse_img = cv2.imread(fuse_path)
        body_mask = cv2.imread(body_mask_path, cv2.IMREAD_GRAYSCALE) > 127  # 转为布尔掩码
        wing_mask = cv2.imread(wing_mask_path, cv2.IMREAD_GRAYSCALE) > 127  # 转为布尔掩码
        
        # 确保掩码与融合图像尺寸一致
        h, w = fuse_img.shape[:2]
        body_mask = cv2.resize(body_mask.astype(np.uint8), (w, h)) > 0
        wing_mask = cv2.resize(wing_mask.astype(np.uint8), (w, h)) > 0
        
        # 定义颜色 (BGR格式，因为OpenCV默认BGR)
        COLOR_BODY = (0, 0, 255)    # 红
        COLOR_WING = (0, 255, 255)  # 黄
        COLOR_BG = (0, 0, 0)        # 黑
        
        # 找到融合图像中的黑色区域（需要补充的区域）
        bg_mask = np.all(fuse_img == COLOR_BG, axis=-1)
        
        # 计算候选区域大小
        red_candidate_mask = bg_mask & body_mask
        yellow_candidate_mask = bg_mask & wing_mask
        
        red_candidate_count = np.sum(red_candidate_mask)
        yellow_candidate_count = np.sum(yellow_candidate_mask)
        
        # 确定是否需要补充
        supplement_red = red_candidate_count <= 10000
        supplement_yellow = yellow_candidate_count <= 10000
        
        # 应用补充
        if supplement_red:
            fuse_img[red_candidate_mask] = COLOR_BODY
        
        if supplement_yellow:
            fuse_img[yellow_candidate_mask] = COLOR_WING
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存结果
        cv2.imwrite(output_path, fuse_img)
        
        # 打印调试信息
        print(f"处理完成: {fuse_path} | 红色候选: {red_candidate_count}, 黄色候选: {yellow_candidate_count} | "
              f"补充红色: {'是' if supplement_red else '否'}, 补充黄色: {'是' if supplement_yellow else '否'}")
        return True
        
    except Exception as e:
        print(f"处理 {fuse_path} 时出错: {str(e)}")
        return False

def process_model(model_name, body_mask_root, wing_mask_root, fuse_root, output_root):
    """处理单个模型的所有视图"""
    # 构建路径
    body_folder = os.path.join(body_mask_root, model_name)
    wing_folder = os.path.join(wing_mask_root, model_name)
    fuse_folder = os.path.join(fuse_root, model_name)
    output_folder = os.path.join(output_root, model_name)
    
    # 检查源文件夹是否存在
    if not os.path.exists(fuse_folder):
        print(f"跳过模型 {model_name}: 融合文件夹不存在")
        return 0
    
    # 获取所有融合视图
    processed_count = 0
    for view_file in os.listdir(fuse_folder):
        if not view_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        fuse_path = os.path.join(fuse_folder, view_file)
        view_name = os.path.splitext(view_file)[0]
        # 假设掩码文件名格式与参考代码一致
        body_mask_path = os.path.join(body_folder, f"{view_name}_wing_part_mask.png")
        wing_mask_path = os.path.join(wing_folder, f"{view_name}_wing_part_mask.png")
        output_path = os.path.join(output_folder, view_file)
        
        if process_view(fuse_path, body_mask_path, wing_mask_path, output_path):
            processed_count += 1
    
    return processed_count

def process_all_models(body_mask_root, wing_mask_root, fuse_root, output_root):
    """处理所有模型"""
    # 获取所有模型（基于融合文件夹）
    model_folders = [d for d in os.listdir(fuse_root) 
                    if os.path.isdir(os.path.join(fuse_root, d))]
    
    print(f"发现 {len(model_folders)} 个模型需要处理")
    
    total_views = 0
    with ThreadPoolExecutor() as executor:
        futures = []
        for model_name in model_folders:
            futures.append(
                executor.submit(
                    process_model,
                    model_name,
                    body_mask_root,
                    wing_mask_root,
                    fuse_root,
                    output_root
                )
            )
        
        # 等待并收集结果
        for future in as_completed(futures):
            views_processed = future.result()
            total_views += views_processed
            print(f"模型处理完成: {views_processed} 个视图")
    
    print(f"\n所有模型处理完成！总共处理 {total_views} 个视图")
    print(f"输出结果保存在: {output_root}")

if __name__ == "__main__":
    # 配置路径 - 修改这些路径以适应您的实际结构
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 输入文件夹（使用掩码文件夹替代原始图像文件夹）
    BODY_MASK_ROOT = os.path.join(SCRIPT_DIR, "body_masks")
    WING_MASK_ROOT = os.path.join(SCRIPT_DIR, "wing_masks")
    FUSE_ROOT = os.path.join(SCRIPT_DIR, "fused_body_wing_fixpoint")
    
    # 输出文件夹
    OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "fixed_fixpoint")
    
    # 开始处理
    process_all_models(BODY_MASK_ROOT, WING_MASK_ROOT, FUSE_ROOT, OUTPUT_ROOT)