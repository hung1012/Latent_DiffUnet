import numpy as np
from dataset.isic_dataloader import get_loader_isic
from dataset.polyp_dataloader import get_loader_polyp
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from models.basic_unet_denose import BasicUNetDe
from models.basic_unet import BasicUNetEncoder
from models.mix_transformer import MixVisionTransformer, mit_b0
import argparse
from functools import partial
from monai.losses.dice import DiceLoss
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
set_determinism(123)
import os

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
from torchvision.transforms.functional import to_pil_image



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

class PolypTrainer(Trainer):
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

        self.best_loss = 1e6
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                  warmup_epochs=100,
                                                  max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def training_step(self, batch):
        image, mask = self.get_input(batch)

        x_latent = self.vae_model.encode(mask).sample()  # latent sample
        # x_start = (x_start) * 2 - 1
        x_t, t, noise = self.model(x=x_latent, pred_type="q_sample")

        pred_latent = self.model(x=x_t, step=t, image=image, pred_type="denoise")  # predicted latent output

        # pred_mask = self.vae_model.decode(pred_latent)

        # loss_dice = self.dice_loss(pred_mask, mask)
        # loss_bce = self.bce(pred_mask, mask)

        # pred_mask = torch.sigmoid(pred_mask)
        # loss_mse = self.mse(pred_mask, mask)

        loss = self.mse(pred_latent, x_latent)

        # loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss, step=self.global_step)

        return loss 
 
    def get_input(self, batch):
        image = batch["image"]
        mask = batch["mask"]
        mask = mask.float()
        return image, mask

    def validation_step(self, batch):
        image, mask = self.get_input(batch)   
        
        output_latent = self.window_infer(image, self.model, pred_type="ddim_sample")

        gt_latent = self.vae_model.encode(mask).sample()  # latent sample

        # output_mask = self.vae_model.decode(output_latent)

        # output = (output > 0.5).float().cpu().numpy()
        # target = mask.cpu().numpy()
        # dice_score = []
        # for i in range(output.shape[1]): #each channels, in case the number of channels is equal to 1
        #     o = output[:, i]
        #     t = target[:, i]
        #     dice_score.append(dice(o, t))

        # loss_dice = self.dice_loss(output_mask, mask)
        # loss_bce = self.bce(output_mask, mask)

        # output_mask = torch.sigmoid(output_mask)
        # loss_mse = self.mse(output_mask, mask)

        # loss = loss_dice + loss_bce + loss_mse

        loss = self.mse(output_latent, gt_latent)

        return loss

    def validation_end(self, mean_val_outputs):
        loss = mean_val_outputs
        self.log("valid_loss", loss, step=self.epoch)

        if loss < self.best_loss:
            self.best_loss = loss
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{loss:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{loss:.4f}.pt"), 
                                        delete_symbol="final_model")

        print(f"valid loss is {loss}")


if __name__ == "__main__":
    data_dir = "/home/admin_mcn/minhtx/data/isic/image"
    logdir = "/mnt/minhtx/logs_isic/ldm"
    model_save_path = os.path.join(logdir, "model")

    env = "pytorch" # or env = "pytorch" if you only have one gpu.

    max_epoch = 100
    batch_size = 8
    val_every = 10
    num_gpus = 1
    device = "cuda:1"

    number_modality = 3
    number_targets = 32


    os.chdir("/home/admin_mcn/hungvq/stable_diffusion")

    config_path = '/mnt/minhtx/VAE/2023-06-07T11-32-09-project.yaml'
    ckpt_path = '/mnt/minhtx/VAE/last.ckpt'
    

    train_ds, val_ds,test_ds = get_loader_isic(data_dir=data_dir, batch_size=batch_size, fold=0)
    
    trainer = PolypTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17750,
                            training_script=__file__)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
