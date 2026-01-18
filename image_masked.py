#这个代码和image_version.py对应我们论文中的Image Prompt Supporting
"""
Hidden Illusion Runner (Masked Version)
功能升级：
自动检测并隔离纯色背景（如白底照片），只保留主体的结构约束。
背景区域将完全交给 Text Prompt 自由发挥。
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import re

import rp
import source.new_stable_diffusion as sd 
from source.learnable_textures import LearnableImageFourier

def parse_args():
    p = argparse.ArgumentParser(description="Generate a Hidden Illusion image.")
    
    # --- 核心输入 ---
    p.add_argument('--target-image', type=str, required=True, help='Path to the visual target image (e.g., lion.png)')
    p.add_argument('--prompt', type=str, default='A medieval coastal town built on a hill, highly detailed architecture, realistic style, sunny day, blue sky', help='Text prompt for the texture')
    p.add_argument('--negative-prompt', type=str, default='blurry, ugly, low quality, watermark, text, white background, simple background')
    
    # --- 背景剔除参数 (新增) ---
    p.add_argument('--bg-threshold', type=float, default=0.9, help='背景剔除阈值 (0.0-1.0)。亮度高于此值的像素被视为背景。白底图建议 0.9-0.95，非白底图设为 1.0 关闭此功能。')
    
    # --- 权重控制 ---
    p.add_argument('--text-strength', type=float, default=1.0, help='SDS Loss 权重')
    p.add_argument('--structure-strength', type=float, default=1.2, help='Latent MSE 权重')
    
    # --- 训练设置 ---
    p.add_argument('--num-iter', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--model-name', type=str, default='./weights/sd-v1-4')
    p.add_argument('--save-dir', type=str, default='outputs/hidden_illusion_masked')
    p.add_argument('--display-interval', type=int, default=50)
    
    return p.parse_args()

def make_save_folder(base):
    os.makedirs(base, exist_ok=True)
    name = time.strftime('%Y%m%d_%H%M%S')
    folder = os.path.join(base, name)
    os.makedirs(folder, exist_ok=True)
    return folder

def is_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config: return False
    except: return False
    return True

def main():
    args = parse_args()

    # 1. 初始化
    gpu = rp.select_torch_device()
    print(f'Using device: {gpu}')
    s = sd.StableDiffusion(gpu, args.model_name)
    
    # 2. 准备 Target
    print(f"Loading target image: {args.target_image}")
    raw_target_img = rp.load_image(args.target_image)
    raw_target_img = rp.resize_image(raw_target_img, (args.size, args.size))
    # [1, 3, H, W] range [0, 1]
    target_img_tensor = rp.as_torch_image(raw_target_img).to(device=s.device, dtype=torch.float32)[None]

    # ==========================================
    # 【新增功能】生成背景蒙版 (Mask Generation)
    # ==========================================
    if args.bg_threshold < 1.0:
        print(f"Applying background masking with threshold {args.bg_threshold}...")
        # 简单逻辑：如果 RGB 三个通道都大于阈值，则认为是白色背景
        # is_bg: [1, 1, H, W]
        is_bg = (target_img_tensor > args.bg_threshold).all(dim=1, keepdim=True)
        
        # mask: 主体为 1，背景为 0
        mask_pixel = (~is_bg).float()
        
        # 将 Mask 缩放到 Latent 尺寸 (512 -> 64)
        # 使用 'nearest' 保持边缘硬度，或者 'bilinear' 获得软边缘
        latent_h, latent_w = args.size // 8, args.size // 8
        mask_latent = F.interpolate(mask_pixel, size=(latent_h, latent_w), mode='bilinear')
        
        # 扩展通道数以匹配 Latents [1, 4, 64, 64]
        mask_latent = mask_latent.repeat(1, 4, 1, 1).to(s.device)
        
        # 保存一下 Mask 方便检查
        # mask_pixel 的 shape 为 [1,1,H,W]，rp.save_image 要求 HWC 或 [C,H,W] 形式且通道>=3
        # 所以在保存前把单通道扩展为 3 通道并搬到 CPU
        mask_vis = mask_pixel.cpu().repeat(1, 3, 1, 1)  # [1,3,H,W]
        rp.save_image(rp.as_numpy_image(mask_vis[0]), "debug_mask.png")
        print("Mask generated and saved to debug_mask.png")
    else:
        # 全 1 掩码 (不剔除背景)
        mask_latent = torch.ones(1, 4, args.size // 8, args.size // 8).to(s.device)

    # 编码 Target 为 Latents
    print("Encoding target image to latents...")
    with torch.no_grad():
        target_latents = s.encode_imgs(target_img_tensor)
        
    # 3. 准备 Text
    print(f"Processing prompt: {args.prompt}")
    text_embeddings = s.get_text_embeddings(args.prompt) 

    # 4. 初始化图像
    print("Creating learnable image...")
    learnable_image = LearnableImageFourier(
        height=args.size, width=args.size, num_features=256, hidden_dim=256, scale=5.0
    ).to(s.device)

    optim = torch.optim.SGD(learnable_image.parameters(), lr=args.lr)

    print(f"Starting optimization for {args.num_iter} steps...")

    # 创建保存文件夹
    target_name = os.path.splitext(os.path.basename(args.target_image))[0]
    raw_snippet = args.prompt[:12].replace(' ', '_')
    snippet = re.sub(r'[^0-9A-Za-z_\u4e00-\u9fff-]', '', raw_snippet)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(args.save_dir, f"{target_name}_{snippet}_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    try:
        for i in range(args.num_iter):
            optim.zero_grad()
            
            # 获取当前生成的图像
            current_image = learnable_image() # [3, H, W]
            pred_rgb = current_image[None]    # [1, 3, H, W]

            # =================================================
            # Pass 1: Text Guidance (SDS)
            # =================================================
            if args.text_strength > 0:
                s.train_step(
                    text_embeddings=text_embeddings,
                    pred_rgb=pred_rgb,
                    target=None,
                    noise_coef=args.text_strength,
                    latent_coef=0,
                    guidance_scale=100
                )

            # =================================================
            # Pass 2: Structure Guidance (Masked Target)
            # =================================================
            if args.structure_strength > 0:
                # 【核心修改逻辑】
                # 为了只让主体产生梯度，背景不产生梯度：
                # 我们构造一个 dynamic_target，它在背景区域等于 current_latents (差值为0)，
                # 在主体区域等于 target_latents (差值为真)。
                
                # 1. 必须先编码当前的图片拿到 Current Latents
                # (虽然增加了一次编码开销，但是不修改 new_stable_diffusion.py 的唯一办法)
                curr_latents = s.encode_imgs(pred_rgb)
                
                # 2. 混合 Target
                # Mask=1 (主体): 使用 target_latents
                # Mask=0 (背景): 使用 curr_latents
                mixed_target = target_latents * mask_latent + curr_latents * (1 - mask_latent)
                
                # 3. 拼接双倍 (满足 chunk(2) 要求)
                doubled_target = torch.cat([mixed_target, mixed_target], dim=0)
                
                # 4. 传入 train_step
                s.train_step(
                    text_embeddings=text_embeddings,
                    pred_rgb=pred_rgb,
                    target=doubled_target, 
                    noise_coef=0,
                    latent_coef=args.structure_strength,
                    guidance_scale=0 
                )

            optim.step()

            # --- 显示进度 ---
            if i % args.display_interval == 0:
                print(f"Step {i}/{args.num_iter} complete")
                if i % 200 == 0:
                    rp.save_image(rp.as_numpy_image(current_image), os.path.join(run_folder, f"progress_{i}.png"))
                
                if is_notebook():
                    with torch.no_grad():
                        np_img = rp.as_numpy_image(current_image)
                        from IPython.display import clear_output
                        clear_output(wait=True)
                        rp.display_image(np_img)

    except KeyboardInterrupt:
        print("Optimization interrupted.")

    print(f"Saving to {run_folder}")
    final_img = rp.as_numpy_image(learnable_image())
    rp.save_image(final_img, os.path.join(run_folder, "final_illusion.png"))
    rp.save_image(raw_target_img, os.path.join(run_folder, "target_reference.png"))

if __name__ == '__main__':
    main()