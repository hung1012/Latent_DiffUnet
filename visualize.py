import os
import torch
import cv2
import numpy as np
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from stablediffusion.ldm.util import instantiate_from_config

os.chdir("/home/admin_mcn/hungvq/stable_diffusion")

#--------------------------load model--------------------------
config_path = '/mnt/minhtx/VAE/2023-06-07T11-32-09-project.yaml'
ckpt_path = '/mnt/minhtx/VAE/last.ckpt'

config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model)

ckpt = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(ckpt['state_dict'])


#--------------------------load data + save test images--------------------------
trans = Compose([ToTensor(), Resize((384,384))])
IMAGE_DIR = "/home/admin_mcn/thaotlp/data/ISIC/image"
MASK_DIR = "/home/admin_mcn/thaotlp/data/ISIC/mask"
OUTPUT_DIR = "/home/admin_mcn/thaotlp/output/test"

LATENT_DECODE_DIR = "/home/admin_mcn/thaotlp/output/latent_gt"
LATENT_GT_DIR = "/home/admin_mcn/hungvq/data/latent_gt"
LATENT_OUTPUT_DIR = "/mnt/thaotlp/output/test"

for filename in os.listdir(LATENT_OUTPUT_DIR):
    fileidx = filename.split(".")[0]
    latent = torch.Tensor(np.load(os.path.join(LATENT_GT_DIR, filename)))

    # reconstruct mask output from latent output
    output = model.decode(latent)
    image = to_pil_image(torch.clamp(output[0].squeeze(0),min=0,max=1), mode='L')
    image.save(os.path.join(LATENT_DECODE_DIR, fileidx + ".png"))

