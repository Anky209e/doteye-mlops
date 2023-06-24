import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from model import UNET
from utils import image_convert,mask_convert,load_checkpoint

model = UNET(in_channels=3,out_channels=1)
CHECKPOINT = "my_checkpoint.pth.tar"
trans_to_tensor = transforms.ToTensor()
trans_to_image = transforms.ToPILImage()
load_checkpoint(torch.load(CHECKPOINT),model)

def segment_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = np.array(img.resize((128,128))).astype(np.float32)
    img = trans_to_tensor(img)
    img = img.unsqueeze(0)
    preds = model(img)
    preds = torch.sigmoid(preds)
    mask_img = trans_to_image(preds[0])
    mask_img.save("out.png")

    return mask_img


if __name__=="__main__":
    img_path = "stage1_test/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732/images/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732.png"

    segment_image(img_path)