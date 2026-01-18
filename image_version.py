#这个代码和image_mask.py对应我们论文中的Image Prompt Supporting
"""
Hidden Illusion Runner (Final Fixed Version)
修复记录：
1. [Logic] 解决了 new_stable_diffusion.py 的 assert 断言冲突（分步计算）。
2. [Type] 解决了 Input type (double) 报错（显式转换为 float32）。
3. [Dim] 解决了 .chunk(2) 维度报错（对 target 进行双倍拼接）。
4. [Attr] 解决了 rp.is_notebook 报错（移除了对 rp 库该函数的依赖）。
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import re

import rp
# 假设你把新版 SD 文件保存为了 source/new_stable_diffusion.py
import source.new_stable_diffusion as sd 
from source.learnable_textures import LearnableImageFourier

def parse_args():
    p = argparse.ArgumentParser(description="Generate a Hidden Illusion image.")
    
    # --- 核心输入 ---
    p.add_argument('--target-image', type=str, required=True, help='Path to the visual target image (e.g., lion.png)')
    p.add_argument('--prompt', type=str, default='A medieval coastal town built on a hill, highly detailed architecture, oil painting style, sunny day, blue sky', help='Text prompt for the texture')
    p.add_argument('--negative-prompt', type=str, default='blurry, ugly, low quality, watermark, text')
    
    # --- 权重控制 ---
    p.add_argument('--text-strength', type=float, default=1.0, help='SDS Loss 权重: 控制图像像"文字描述"的程度')
    p.add_argument('--structure-strength', type=float, default=1.2, help='Latent MSE 权重: 控制图像像"目标图片"的程度')
    
    # --- 训练设置 ---
    p.add_argument('--num-iter', type=int, default=1200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--model-name', type=str, default="./weights/sd-v1-4")
    p.add_argument('--save-dir', type=str, default='outputs/hidden_illusion')
    
    # --- 显示设置 ---
    p.add_argument('--display-interval', type=int, default=50)
    
    return p.parse_args()

def make_save_folder(base):
    os.makedirs(base, exist_ok=True)
    name = time.strftime('%Y%m%d_%H%M%S')
    folder = os.path.join(base, name)
    os.makedirs(folder, exist_ok=True)
    return folder

# 一个不依赖 rp 库的 Notebook 检测函数
def is_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

def main():
    args = parse_args()

    # 1. 初始化
    gpu = rp.select_torch_device()
    print(f'Using device: {gpu}')
    s = sd.StableDiffusion(gpu, args.model_name)
    
    # 2. 准备 Target (狮子)
    print(f"Loading target image: {args.target_image}")
    raw_target_img = rp.load_image(args.target_image)
    raw_target_img = rp.resize_image(raw_target_img, (args.size, args.size))
    target_img_tensor = rp.as_torch_image(raw_target_img).to(device=s.device, dtype=torch.float32)[None]

    
    # 编码为 Latents (结构基准)
    print("Encoding target image to latents...")
    with torch.no_grad():
        target_latents = s.encode_imgs(target_img_tensor)
        
    # 3. 准备 Text (山)
    print(f"Processing prompt: {args.prompt}")
    text_embeddings = s.get_text_embeddings(args.prompt) 

    # 4. 初始化图像
    print("Creating learnable image...")
    # scale 参数决定了初始的高频细节丰富程度
    learnable_image = LearnableImageFourier(
        height=args.size, width=args.size, num_features=256, hidden_dim=256, scale=5.0
    ).to(s.device)

    optim = torch.optim.SGD(learnable_image.parameters(), lr=args.lr)

    print(f"Starting optimization for {args.num_iter} steps...")

    # --- 为本次运行创建专用保存文件夹 ---
    # 以目标图片名 + prompt 前若干字作为文件夹名，追加时间戳保证唯一
    target_name = os.path.splitext(os.path.basename(args.target_image))[0]
    # 取 prompt 前 12 个字符，移除不安全字符并替换空格
    raw_snippet = args.prompt[:12].replace(' ', '_')
    snippet = re.sub(r'[^0-9A-Za-z_\u4e00-\u9fff-]', '', raw_snippet)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(args.save_dir, f"{target_name}_{snippet}_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    # ----------------------------------------

    try:
        for i in range(args.num_iter):
            
            optim.zero_grad()
            
            # 获取当前图像
            current_image = learnable_image() # [3, H, W]
            pred_rgb = current_image[None]    # [1, 3, H, W]

            # =================================================
            # 第一步：计算"纹理" Loss (Text Guidance / SDS)
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
            # 第二步：计算"结构" Loss (Visual Guidance / Regression)
            # =================================================
            if args.structure_strength > 0:
                # 【修复点 2】：双倍拼接 target 以满足 chunk(2) 的维度要求
                doubled_target = torch.cat([target_latents, target_latents], dim=0)
                
                s.train_step(
                    text_embeddings=text_embeddings, # 占位
                    pred_rgb=pred_rgb,
                    target=doubled_target, # 使用双倍 target
                    noise_coef=0,
                    latent_coef=args.structure_strength,
                    guidance_scale=0 
                )

            # =================================================
            # 更新参数
            # =================================================
            optim.step()

            # --- 显示进度 ---
            if i % args.display_interval == 0:
                print(f"Step {i}/{args.num_iter} complete")

                # 使用自定义的 is_notebook 函数，或者直接跳过
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

    # 保存最终结果到本次运行文件夹
    print(f"Saving to {run_folder}")
    final_img = rp.as_numpy_image(learnable_image())
    rp.save_image(final_img, os.path.join(run_folder, "final_illusion.png"))

    # 保存参考图
    rp.save_image(raw_target_img, os.path.join(run_folder, "target_reference.png"))

if __name__ == '__main__':
    main()