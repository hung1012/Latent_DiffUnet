import numpy as np
# from dataset.brats_data_utils_multi_label import get_loader_brats
from dataset.isic_dataloader import get_loader_isic
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
import argparse
from monai.losses.dice import DiceLoss
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
set_determinism(123)
import os
from ldm.models.autoencoder import AutoencoderKL
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config



class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        

        self.embed_model = BasicUNetEncoder(2, number_modality, number_targets, [32, 32, 64, 64, 128, 128, 256, 512, 32])

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
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 12, 12), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out

class ISICTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[384, 384],
                                                sw_batch_size=1,
                                                overlap=0.25)
        self.model = DiffUNet()

        print('Start creating and loading vae model ...')
        config_path = '/home/admin_mcn/hungvq/stable_diffusion/logs/2023-06-07T11-32-09_seg_diff_autoencoder/configs/2023-06-07T11-32-09-project.yaml'
        ckpt_path = '/home/admin_mcn/hungvq/stable_diffusion/logs/2023-06-07T11-32-09_seg_diff_autoencoder/checkpoints/last.ckpt'
        config = OmegaConf.load(config_path)
        vae_model = instantiate_from_config(config.model)
        vae_model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
        self.vae = vae_model.eval().to(device)
        for name, param in self.vae.named_parameters():
            param.requires_grad = False

        self.best_mean_dice = 1e4
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                  warmup_epochs=100,
                                                  max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def training_step(self, batch):
        image, latent_mask, mask = self.get_input(batch)
        x_start = latent_mask

        x_start = (x_start) * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        

        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        decode_pred = self.vae.decode(pred_xstart)
        decode_pred = torch.sigmoid(decode_pred)
        loss_dice = self.dice_loss(decode_pred, mask)
        loss_bce = self.bce(decode_pred, mask)
        loss_mse = self.mse(pred_xstart, latent_mask)

        loss = loss_dice + loss_mse + loss_bce

        self.log("train_loss", loss, step=self.global_step)

        return loss 
 
    def get_input(self, batch):
        image = batch["image"]
        latent_mask = batch["latent"]
        mask = batch['mask']
       
   
        mask = mask.float()
        return image, latent_mask, mask

    def validation_step(self, batch):
        image, latent_mask, mask = self.get_input(batch)    
        

        output = self.window_infer(image, self.model, pred_type="ddim_sample")
        decode_pred = self.vae.decode(output)
        decode_pred = torch.sigmoid(decode_pred)
        decode_pred = (decode_pred>0.5).float()

        loss_dice = self.dice_loss(decode_pred, mask)
        loss_bce = self.bce(decode_pred, mask)
        loss_mse = self.mse(output, latent_mask)

        loss = loss_dice + loss_mse + loss_bce
        
        return loss

    def validation_end(self, mean_val_outputs):
        mean_val_outputs = mean_val_outputs.item()

        self.log("valid loss", mean_val_outputs, step=self.epoch)

        if mean_val_outputs < self.best_mean_dice:
            self.best_mean_dice = mean_val_outputs
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_val_outputs:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_val_outputs:.4f}.pt"), 
                                        delete_symbol="final_model")

        print(f"valid loss is {mean_val_outputs}")

if __name__ == "__main__":

    data_dir = "/home/admin_mcn/thaotlp/data/ISIC/image"
    mask_dir = "/home/admin_mcn/thaotlp/data/ISIC/mask"
    logdir = "/home/admin_mcn/hungvq/logs/latent_12_decode"
    model_save_path = os.path.join(logdir, "model")

    env = "pytorch" # or env = "pytorch" if you only have one gpu.

    max_epoch = 300
    batch_size = 1
    val_every = 10
    num_gpus = 1
    device = "cuda:1"

    number_modality = 3
    number_targets = 32

    train_ds, val_ds, test_ds = get_loader_isic(data_dir=data_dir, mask_dir=mask_dir, batch_size=batch_size, fold=0)

    
    trainer = ISICTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17750,
                            training_script=__file__)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)


