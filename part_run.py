import os
import sys
import time

import torch
import random
import warnings
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2
from data import NASA_3D 
import numpy as np

sys.path.append(os.path.abspath('../'))
from libs.lib_o3d import batch_geo_feature
from partmodel.post_search import search_prompt, search_vweight
from rendering.prejection import RealisticProjection
from partclip import clip
from libs.lib_vis import get_colored_image_pca_sep

warnings.filterwarnings("ignore")

PC_NUM = 2048

TRANS = -1.5

params = {'vit_b16': {'maxpoolz': 5, 'maxpoolxy': 11, 'maxpoolpadz': 2, 'maxpoolpadxy': 5,
                      'convz': 5, 'convxy': 5, 'convsigmaxy': 1, 'convsigmaz': 2, 'convpadz': 2, 'convpadxy': 2,
                      'imgbias': 0., 'depth_bias': 0.3, 'obj_ratio': 0.7, 'bg_clr': 0.0,
                      'resolution': 224, 'depth': 112}}
net = 'vit_b16'

cat2id = {'satellite': 0}

class Extractor(torch.nn.Module):
    def __init__(self, model):
        super(Extractor, self).__init__()

        self.model = model
        self.pc_views = RealisticProjection(params[net])
        self.get_img = self.pc_views.get_img
        self.params_dict = params[net]

    def mv_proj(self, pc):
        img, is_seen, point_loc_in_img = self.get_img(pc)
        img = img[:, :, 20:204, 20:204]
        point_loc_in_img = torch.ceil((point_loc_in_img - 20) * 224. / 184.)
        img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=True)
        return img, is_seen, point_loc_in_img

    def forward(self, pc, is_save=False):
        img, is_seen, point_loc_in_img = self.mv_proj(pc)

        _, x = self.model.encode_image(img)
        x = x / x.norm(dim=-1, keepdim=True)
        B, L, C = x.shape
        if is_save:
            feats = torch.nn.functional.interpolate(x.reshape(B, 14, 14, C).permute(0, 3, 1, 2), size=(224, 224),
                                                    mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            for i in range(len(img)):
                # Normalize the depth values to the desired range (0 to 255 in this example)
                normalized_depth = cv2.normalize(img[i][0].cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)

                # Convert the depth matrix to an 8-bit unsigned integer (uint8) image
                depth_image = np.uint8(normalized_depth)
                feat = feats[i]
                get_colored_image_pca_sep(feat.cpu().numpy(), i)
                cv2.imwrite(f'saved_depth_{i}.png', depth_image)

        x = x.reshape(B, 14, 14, C).permute(0, 3, 1, 2)
        return is_seen, point_loc_in_img, x

def extract_feature_maps(model_name, data_path, class_choice, device):
    model, _ = clip.load(model_name, device=device)
    model.to(device)

    segmentor = Extractor(model)
    segmentor = segmentor.to(device)
    segmentor.eval()

    output_path = 'output/{}/'.format(model_name.replace('/', '_'))
    mode = 'test'

    save_path = os.path.join(output_path, class_choice)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('\nStart to extract and save feature maps of class {}...'.format(class_choice))
    test_loader = DataLoader(
        NASA_3D(num_points=2048, partition=mode), batch_size=1, shuffle=False, drop_last=False)

    # 存储样本名称的列表
    name_list = []
    
    # 遍历每个样本单独处理
    for batch in tqdm(test_loader):
        pc = batch[0]
        label = batch[1]
        pc, label = pc.cuda(), label.cuda()
        name = batch[2][0]  # 提取文件名字符串
        sample_save_path = os.path.join(save_path, name)
        name_list.append(name)
        
        if os.path.exists(os.path.join(sample_save_path, "features.pt")):
            continue
            
        os.makedirs(sample_save_path, exist_ok=True)
        
        with torch.no_grad():
            is_seen, point_loc_in_img, feat = segmentor(pc)
            normal, fpfh = batch_geo_feature(pc, voxel_size=0.05)
            
            # Save features for each sample individually
            torch.save(pc, os.path.join(sample_save_path, "pc.pt"))
            torch.save(normal, os.path.join(sample_save_path, "normal.pt"))
            torch.save(fpfh, os.path.join(sample_save_path, "fpfh.pt"))
            torch.save(feat, os.path.join(sample_save_path, "features.pt"))
            torch.save(label.squeeze(), os.path.join(sample_save_path, "labels.pt"))
            torch.save(is_seen, os.path.join(sample_save_path, "ifseen.pt"))
            torch.save(point_loc_in_img, os.path.join(sample_save_path, "pointloc.pt"))

    # 保存样本名称列表到txt文件
    name_file_path = os.path.join(save_path, "names.txt")
    with open(name_file_path, 'w') as f:
        for name in name_list:
            f.write(name + '\n')
    print(f'Saved sample names to {name_file_path}')

def main(args):
    random.seed(0)
    device = args.device
    model_name = args.modelname

    data_path = args.datasetpath
    only_evaluate = args.onlyevaluate
    class_choice = args.classchoice
    # 提取并保存特征图
    extract_feature_maps(model_name, data_path, class_choice, device)

    start_time = time.time()
    # 测试或搜索提示和视图权重
    prompts = search_prompt(class_choice, model_name, only_evaluate=only_evaluate,img_size=(params[net]['resolution'], params[net]['resolution']))
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time * 1000:.2f} ms")
    if not only_evaluate:
        search_vweight(class_choice, model_name, prompts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default='ViT-B/16')
    parser.add_argument('--classchoice', default='satellite')
    parser.add_argument('--datasetpath', default='data')
    parser.add_argument('--onlyevaluate', default=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    main(args)
    