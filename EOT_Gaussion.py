#%%
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
from scipy import signal
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_id = "timbrooks/instruct-pix2pix"
pretrained_model_name_or_path = model_id
torch.cuda.device_count()
from torchvision import transforms
from pathlib import Path
from Functions import *

import pandas as pd
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = './instruct-pix2pix-main/diffuser_cache'
cop_path = './instruct-pix2pix-main/cop_file'
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained('timbrooks/instruct-pix2pix', torch_dtype=torch.float16,
                                                 safety_checker=None, cache_dir=model_path, local_files_only=True)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)



#%%
pgd_alpha = 1 / 255
pgd_eps = 0.05
max_steps = 30
center_crop = False

#%%
# i = 0
def gaussian_kernel(size: int, sigma: float):
    """Generates a 2D Gaussian kernel."""
    gkern1d = torch.from_numpy(np.outer(signal.gaussian(size, sigma), signal.gaussian(size, sigma)))
    gkern2d = gkern1d.float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, size, size]
    return gkern2d


def apply_gaussian_smoothing(input: torch.Tensor, kernel_size: int, sigma: float):
    """Applies Gaussian smoothing on a tensor."""
    # Create the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)

    # Apply the kernel to each channel separately
    smoothed_channels = []
    for c in range(input.shape[1]):  # Loop over the channels
        channel = input[:, c:c + 1, :, :]  # Shape: [B, 1, H, W]
        padding = kernel_size // 2
        smoothed_channel = F.conv2d(channel, kernel, padding=padding)
        smoothed_channels.append(smoothed_channel)

    # Concatenate all channels back
    smoothed = torch.cat(smoothed_channels, dim=1)  # Shape: [B, C, H, W]
    return smoothed

if not os.path.exists('./instruct-pix2pix-main/002_Data/EOT-G/Gen'):
    os.mkdir('./instruct-pix2pix-main/002_Data/EOT-G/Gen')
if not os.path.exists('./instruct-pix2pix-main/002_Data/EOT-G/Adv'):
    os.mkdir('./instruct-pix2pix-main/002_Data/EOT-G/Adv')
if not os.path.exists('./instruct-pix2pix-main/002_Data/EOT-G/AdvGen'):
    os.mkdir('./instruct-pix2pix-main/002_Data/EOT-G/AdvGen')
if not os.path.exists('./instruct-pix2pix-main/002_Data/EOT-G/Ori'):
    os.mkdir('./instruct-pix2pix-main/002_Data/EOT-G/Ori')
if not os.path.exists('./instruct-pix2pix-main/002_Data/EOT-G/EXCEL'):
    os.mkdir('./instruct-pix2pix-main/002_Data/EOT-G/EXCEL')

root_path = './instruct-pix2pix-main/train_data'
train_data_path = os.listdir('./instruct-pix2pix-main/train_data')

for i in range(len(train_data_path)):
    print(train_data_path[i])
    save_ori = os.path.join('./instruct-pix2pix-main/002_Data/EOT-G/Ori', train_data_path[i])
    if not os.path.exists(save_ori):
        os.mkdir(save_ori)
    save_gen = os.path.join('./instruct-pix2pix-main/002_Data/EOT-G/Gen', train_data_path[i])
    if not os.path.exists(save_gen):
        os.mkdir(save_gen)
    save_adv = os.path.join('./instruct-pix2pix-main/002_Data/EOT-G/Adv', train_data_path[i])
    if not os.path.exists(save_adv):
        os.mkdir(save_adv)
    save_advgen = os.path.join('./instruct-pix2pix-main/002_Data/EOT-G/AdvGen', train_data_path[i])
    if not os.path.exists(save_advgen):
        os.mkdir(save_advgen)
    save_result = os.path.join('./instruct-pix2pix-main/002_Data/EOT-G/EXCEL', train_data_path[i])
    if not os.path.exists(save_result):
        os.mkdir(save_result)
    print('Make Dir Done!')


    image_path = os.path.join(root_path, train_data_path[i])
    image_list = os.listdir(image_path)
    resolution = 512
    with open(os.path.join(image_path, 'prompt.json'), 'r', encoding='utf-8') as f:
        load_json = json.load(f)
    prompt = load_json['edit']
    name_list = []
    sim_image_bef_list = []
    sim_image_aft_list = []
    sim_image_adv_list = []

    for j in range(len(image_list)):
        if image_list[j].endswith('_0.jpg'):
            # print('hh')
            input_path = os.path.join(image_path, image_list[j])
            name_list.append(image_list[j])
            perturbed_data = load_data(input_path, resolution, center_crop=False)  # 初始img  可更新
            # print(perturbed_data.shape)
            kernel_size = 5
            sigma = 1.5

            gaussian_data = apply_gaussian_smoothing(perturbed_data, kernel_size, sigma)
            gaussian_data = torch.clamp(gaussian_data, min=0, max=1)
            tgt_data = load_data(input_path, resolution, center_crop=False)

            original_data = perturbed_data.clone()  # 初始img
            aaa = original_data.detach().cpu().numpy()[0]
            plt.imsave(os.path.join(save_ori, image_list[j]), aaa.transpose(1, 2, 0))
            generator = torch.Generator("cuda").manual_seed(33)
            images = \
            pipe(prompt, image=Image.open(input_path).resize((512, 512)), num_inference_steps=100, image_guidance_scale=1.2,
                 generator=generator).images[0]
            images.save(os.path.join(save_gen, image_list[j]))
            original_images = original_data
            perturbed_images = perturbed_data.detach().clone()
            tgt_images = tgt_data.detach().clone()
            tgt_emb = get_emb(tgt_images).detach().clone()

            optimizer = optim.Adam([perturbed_images])
            for step in range(max_steps):
                perturbed_images.requires_grad = True
                img_emb = get_emb(perturbed_images)
                optimizer.zero_grad()

                vae.zero_grad()
                loss = -F.mse_loss(img_emb.float(), tgt_emb.float())

                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    print(f"PGD loss - step {step}, loss: {loss.detach().item()}")
            noised_imgs = perturbed_images.detach().cpu().numpy()[0]
            plt.imsave(os.path.join(save_adv, image_list[j]),
                       np.clip(noised_imgs.transpose(1, 2, 0), 0, 1))

            generator = torch.Generator("cuda").manual_seed(33)
            images = \
            pipe(prompt, image=Image.fromarray(np.uint8(noised_imgs.transpose(1, 2, 0) * 255)), num_inference_steps=100,
                 image_guidance_scale=1.2, generator=generator).images[0]

            images.save(os.path.join(save_advgen, image_list[j]))

            x = np.array(Image.open(os.path.join(save_ori, image_list[j])).resize(
                (512, 512))) / 255  # benign
            x_adv = np.array(
                Image.open(os.path.join(save_adv, image_list[j])).resize(
                    (512, 512))) / 255  # benign
            x_gen = np.array(Image.open(os.path.join(save_gen, image_list[j])).resize(
                (512, 512))) / 255  # ori_xg
            x_gen_attack = np.array(
                Image.open(os.path.join(save_advgen, image_list[j])).resize(
                    (512, 512))) / 255

            clip_similarity = ClipSimilarity().cuda()
            image_features_benign = clip_similarity.encode_image(
                image=torch.tensor(x.transpose(2, 0, 1)).unsqueeze(0).to(device))
            image_features_gen = clip_similarity.encode_image(
                image=torch.tensor(x_gen.transpose(2, 0, 1)).unsqueeze(0).to(device))
            image_feature_adv = clip_similarity.encode_image(
                image=torch.tensor(x_adv.transpose(2, 0, 1)).unsqueeze(0).to(device))
            image_features_attack = clip_similarity.encode_image(
                image=torch.tensor(x_gen_attack.transpose(2, 0, 1)).unsqueeze(0).to(device))
            sim_image_bef = F.cosine_similarity(image_features_benign, image_features_gen)[0]
            sim_image_aft = F.cosine_similarity(image_features_benign, image_features_attack)[0]
            sim_image_adv = F.cosine_similarity(image_features_benign, image_feature_adv)[0]
            sim_image_bef_list.append(sim_image_bef.detach().cpu().numpy())
            sim_image_aft_list.append(sim_image_aft.detach().cpu().numpy())
            sim_image_adv_list.append(sim_image_adv.detach().cpu().numpy())


        else:
            continue
    data = {'file_name': name_list, 'sim_image_bef':sim_image_bef_list, 'sim_image_aft':sim_image_aft_list, 'sim_image_adv':sim_image_adv_list}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_result, 'result.csv'), index=False)