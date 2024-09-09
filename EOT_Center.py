import sys
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from einops import rearrange
import ssl
from tqdm import tqdm
import time
import torch
import os
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers import DiffusionPipeline
import copy
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torch import optim
import json
import random
random.seed(333)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_id = "timbrooks/instruct-pix2pix"
pretrained_model_name_or_path = model_id
torch.cuda.device_count()
from torchvision import transforms
from pathlib import Path
from Functions import *

import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model_path = './instruct-pix2pix-main/diffuser_cache'
# cop_path = './instruct-pix2pix-main/cop_file'
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained('timbrooks/instruct-pix2pix', torch_dtype=torch.float16,
                                                 safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

pgd_alpha = 1 / 255
pgd_eps = 0.05
max_steps = 30
center_crop = False

# i = 0
def center_crop(images, new_height, new_width):
    # Calculate cropping start and end points
    _, _, height, width = images.shape
    startx = width // 2 - (new_width // 2)
    starty = height // 2 - (new_height // 2)
    endx = startx + new_width
    endy = starty + new_height

    # Crop and return
    return images[:, :, starty:endy, startx:endx]

image_dir = "human_data"
if not os.path.exists("perturbed"):
    os.mkdir("perturbed")

for i in range(10):
    image_path = os.path.join(image_dir, f"{i+1}.png")
    perturbed_data = load_data(image_path)
    tgt_data = load_data(image_path)
    original_data = perturbed_data.clone()
    perturbed_images = perturbed_data.detach().clone()
    tgt_images = tgt_data.detach().clone()
    tgt_emb = get_emb(tgt_images).detach().clone()

    optimizer = optim.Adam([perturbed_images])
    from tqdm import trange
    pbar = trange(max_steps)
    for step in pbar:
        perturbed_images.requires_grad = True
        img_emb = get_emb(perturbed_images)
        optimizer.zero_grad()
        vae.zero_grad()
        loss = -F.mse_loss(img_emb.float(), tgt_emb.float())
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.detach().item())
    noised_imgs = perturbed_images.detach().cpu().numpy()[0]
    plt.imsave(f"perturbed/{i+1}.png", np.clip(noised_imgs.transpose(1, 2, 0), 0, 1))