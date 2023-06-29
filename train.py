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
        
        self.cond_model = MixVisionTransformer(img_size=384, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 32],
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
            
        self.model = DiffUNet()

        self.best_loss = 1e6
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-2)
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                  warmup_epochs=100,
                                                  max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def training_step(self, batch):
        image, mask, latent, name = self.get_input(batch)

        x_t, t, noise = self.model(x=latent, pred_type="q_sample")

        pred_latent = self.model(x=x_t, step=t, image=image, pred_type="denoise") 

        loss = self.mse(pred_latent, latent)

        self.log("train_loss", loss, step=self.global_step)

        return loss 
 
    def get_input(self, batch):
        image = batch["image"]
        mask = batch["mask"]
        latent = batch["latent"]
        name = batch["name"]
        mask = mask.float()
        return image, mask, latent, name

    def validation_step(self, batch):
        image, mask, latent, name = self.get_input(batch)   
        
        output_latent = self.window_infer(image, self.model, pred_type="ddim_sample")
        
        loss = self.mse(output_latent, latent)

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
    logdir = "/mnt/thaotlp/logs/logs_isic/0629"
    model_save_path = os.path.join(logdir, "model")

    env = "pytorch" # or env = "pytorch" if you only have one gpu.

    max_epoch = 100
    batch_size = 8
    val_every = 10
    num_gpus = 1
    device = "cuda:0"

    number_modality = 3
    number_targets = 32
    

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
