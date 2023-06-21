import numpy as np
from dataset.brats_data_utils_multi_label import get_loader_brats
from dataset.isic_dataloader import get_loader_isic
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.evaluation.metric import dice, hausdorff_distance_95, recall, fscore
import argparse
import yaml 
import cv2
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
from models.basic_unet_denose import BasicUNetDe
from models.basic_unet import BasicUNetEncoder
from models.mix_transformer import MixVisionTransformer, mit_b0

from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage

set_determinism(123)
import os
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
from torchvision.transforms.functional import to_pil_image




def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.cond_model = MixVisionTransformer(img_size=384, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[32, 64, 64, 32],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[32, 16, 4, 1])

        self.embed_model = BasicUNetEncoder(2, number_targets, number_targets, [128, 256, 512, 32])  ## remove encoder for condition

        self.model = BasicUNetDe(2, number_targets, number_targets, [128, 256, 512, 32], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))
   
        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)


    def forward(self, image=None, x=None, pred_type=None, step=None):

        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            # add conditional encoder
            encoded_image = self.cond_model(image)
            embeddings = self.embed_model(encoded_image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            # add conditional encoder
            encoded_image = self.cond_model(image)
            embeddings = self.embed_model(encoded_image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 12, 12), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out

            

class ISICTester(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[384, 384],
                                                sw_batch_size=1,
                                                overlap=0.25)

        config = OmegaConf.load(config_path)
        self.vae_model = instantiate_from_config(config.model)
        self.vae_model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
        self.vae_model.to(device)
        for param in self.vae_model.parameters():
            param.requires_grad = False
            
        self.model = DiffUNet()

        self.mse = nn.MSELoss()
 
    def get_input(self, batch):
        image = batch["image"]
        mask = batch["mask"]
        name = batch["name"]
        mask = mask.float()
        return image, mask, name

    def validation_step(self, batch):
        image, mask, name = self.get_input(batch)  

        vis_mask = mask.detach().cpu().numpy()[0]
        vis_mask = np.transpose(vis_mask, (2,1,0)) 
        
        output_latent = self.window_infer(image, self.model, pred_type="ddim_sample")

        gt_latent = self.vae_model.encode(mask).sample()  # latent sample

        loss = self.mse(output_latent, gt_latent).item()
        print(loss)

        output_mask = self.vae_model.decode(output_latent).detach().cpu().numpy()[0]
        output_mask = np.transpose(output_mask, (2,1,0))
        print(output_mask.shape)
        cv2.imwrite(os.path.join(output_dir, name + '.png'), output_mask)
        print("save file" + os.path.join(output_dir, name + '.png'))

        return [loss]


if __name__ == "__main__":

    data_dir = "/home/admin_mcn/minhtx/data/isic/image"
    logdir = "/mnt/minhtx/logs_isic/ldm/model/final_model_1.7877.pt"
    output_dir = '/mnt/minhtx/output/isic'


    os.chdir("/home/admin_mcn/hungvq/stable_diffusion")

    config_path = '/mnt/minhtx/VAE/2023-06-07T11-32-09-project.yaml'
    ckpt_path = '/mnt/minhtx/VAE/last.ckpt'

    max_epoch = 300
    batch_size = 4
    val_every = 10
    device = "cuda:1"

    number_modality = 3
    number_targets = 32 

    train_ds, val_ds, test_ds = get_loader_isic(data_dir=data_dir, batch_size=batch_size, fold=0)
    
    tester = ISICTester(env_type="pytorch",
                        max_epochs=max_epoch,
                        batch_size=batch_size,
                        device=device,
                        val_every=val_every,
                        num_gpus=1,
                        master_port=17751,
                        training_script=__file__)

    
    tester.load_state_dict(logdir)
    v_mean = tester.validation_single_gpu(val_dataset=test_ds)

    print(f"v_mean is {v_mean}")

        