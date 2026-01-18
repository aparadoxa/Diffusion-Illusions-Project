import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
# 假设 model.py 在同级目录下，包含 BiSeNet 类
from model import BiSeNet

class FaceParser:
    def __init__(self, model_path='cp/79999_iter.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = 19
        
        self.net = BiSeNet(n_classes=self.n_classes)
        self.net.to(self.device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型权重未找到: {model_path}")
            
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # 定义部位索引 (BiSeNet的标准索引)
        # 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye, 
        # 10: nose, 11: mouth, 12: u_lip, 13: l_lip
        self.parts_definitions = {
            "Left_Eye": [4],
            "Right_Eye": [5],
            "Nose": [10],
            "Mouth": [11, 12, 13],
            "Face_Skin": [1] # 用作盘子轮廓
        }

    def parse(self, img_input, target_size=(512, 512)):
        """
        输入图片，返回统一尺寸的掩码字典
        """
        if isinstance(img_input, str):
            if not os.path.exists(img_input):
                raise FileNotFoundError(f"找不到图片: {img_input}")
            img_pil = Image.open(img_input).convert('RGB')
        elif isinstance(img_input, np.ndarray):
            img_pil = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
        
        # 调整到目标训练尺寸
        img_pil = img_pil.resize(target_size, Image.BILINEAR)
        W, H = img_pil.size
        
        img_tensor = self.to_tensor(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.net(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

        results = {
            "masks": {},
            "original_img": np.array(img_pil)
        }

        # 生成各个部位的二值化Mask (0 or 1)
        for part_name, indices in self.parts_definitions.items():
            mask = np.isin(parsing, indices).astype(np.float32)
            # 去除噪点
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=1)
            results['masks'][part_name] = mask

        # 生成一个 "背景+头发" 的Mask，用于让这些区域保持白色或黑色
        all_features = sum([results['masks'][k] for k in results['masks']])
        results['masks']['Background'] = 1.0 - np.clip(all_features, 0, 1)

        return results