#这个代码对应我们论文中的High-low frequency separation
import torch
import torch.nn.functional as F
import numpy as np
import rp
import tqdm
import os
from PIL import Image
from torchvision.transforms.functional import gaussian_blur
from diffusers import StableDiffusionImg2ImgPipeline

from source.new_stable_diffusion import StableDiffusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "./weights/sd-v1-4"

# 目标图片 (低频/远看)
INPUT_IMAGE_PATH = "images/doge.jpg"

# 提示词 (高频/近看)
#PROMPT = "a magnificent snowy mountain range, alps, matte painting, trending on artstation, 8k, highly detailed, photorealistic"
#PROMPT = "a bustling medieval village with rustic cottages, market stalls, aerial view, highly detailed, fantasy art, warm lighting"
PROMPT = "futuristic cyberpunk city street at night, neon lights, rain reflections, wet pavement, blade runner style, high contrast"
NEGATIVE_PROMPT = "blur, low quality, ugly, flat, low contrast, text, watermark, noise, grainy"

NUM_ITER = 1500       # 迭代次数
LR = 0.05             # 学习率

# 模糊核大小 (越大越需要站得远看)
KERNEL_SIZE = 33      
SIGMA = 2.0

DREAM_WEIGHT = 1.0    
IMAGE_WEIGHT = 10.0 

DREAM_STRENGTH = 0.5  
DREAM_INTERVAL = 10 


class LearnableImage(torch.nn.Module):
    def __init__(self, height=512, width=512):
        super().__init__()
        self.pixels = torch.nn.Parameter(
            torch.full((1, 3, height, width), 0.5).to(device)
        )

    def forward(self):
        return self.pixels.clamp(0, 1)

def load_target_image(path, size=(512, 512)):
    if not os.path.exists(path):
        print(f"错误：找不到 {path}")
        return None
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize(size, Image.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    except Exception as e:
        print(f"加载图片出错: {e}")
        return None


def run_illusion_dtl():
    print(">>> [1/4] 正在加载 Stable Diffusion...")
    sd_base = StableDiffusion(device=device, checkpoint_path=model_path)
    
    print(">>> [2/4] 初始化(Img2Img Pipeline)...")
    dreamer_pipe = StableDiffusionImg2ImgPipeline(
        vae=sd_base.vae,
        text_encoder=sd_base.text_encoder,
        tokenizer=sd_base.tokenizer,
        unet=sd_base.unet,
        scheduler=sd_base.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    ).to(device)
    dreamer_pipe.set_progress_bar_config(disable=True)

    print(f">>> [3/4] 加载目标图片: {INPUT_IMAGE_PATH}")
    target_img_low_freq = load_target_image(INPUT_IMAGE_PATH)
    if target_img_low_freq is None: return

    image_model = LearnableImage().to(device)
    optimizer = torch.optim.Adam(image_model.parameters(), lr=LR)
    
    output_dir = "outputs/doge_cyber"
    rp.make_directory(output_dir)
    print(f">>> [4/4] 开始 DTL 训练... 结果保存至 {output_dir}")

    current_dream_target = None
    
    pbar = tqdm.tqdm(range(NUM_ITER))
    for i in pbar:
        optimizer.zero_grad()
        current_image = image_model()
        
        if i % DREAM_INTERVAL == 0:
            with torch.no_grad():
                # 将当前画布转为 PIL 格式喂给 pipeline
                curr_pil = rp.as_numpy_image(current_image[0])
                curr_pil = (curr_pil * 255).astype(np.uint8)
                curr_pil = Image.fromarray(curr_pil)
                
                # 使用 SDEdit 生成高质量目标
                dream_out = dreamer_pipe(
                    prompt=PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=curr_pil,
                    strength=DREAM_STRENGTH, 
                    guidance_scale=7.5,
                    num_inference_steps=20 
                ).images[0]
                
                dream_np = np.array(dream_out).astype(np.float32) / 255.0
                current_dream_target = torch.from_numpy(dream_np).permute(2, 0, 1).unsqueeze(0).to(device)

        
        # 1. High Freq Loss
        loss_high = F.mse_loss(current_image, current_dream_target)
        
        # 2. Low Freq Loss
        current_blurred = gaussian_blur(current_image, kernel_size=KERNEL_SIZE, sigma=SIGMA)
        loss_low = F.mse_loss(current_blurred, target_img_low_freq)
        
        # 3. 总 Loss
        total_loss = (DREAM_WEIGHT * loss_high) + (IMAGE_WEIGHT * loss_low)
        
        total_loss.backward()
        optimizer.step()
        
        # 可视化 
        if i % 50 == 0:
            pbar.set_description(f"Loss High: {loss_high.item():.4f} | Low: {loss_low.item():.4f}")
            with torch.no_grad():
                vis_row = torch.cat([current_image, current_dream_target, current_blurred], dim=3)
                rp.save_image(vis_row[0], f"{output_dir}/step_{i:04d}.png")

    # 保存
    final = image_model().clamp(0, 1)
    rp.save_image(final[0], f"{output_dir}/final_result.png")
    print("训练完成！")
    print("左侧图：直接观看")
    print("中间图：AI 生成的完美目标 (参考)")
    print("右侧图：眯眼观看效果")

if __name__ == '__main__':
    run_illusion_dtl()