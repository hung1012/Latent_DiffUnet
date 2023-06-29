import os
import json
import math
import numpy as np
import torch
<<<<<<< HEAD
=======
import cv2
>>>>>>> 2d5202bb4e0b38ce8bf4b50e8560894e0271aafa
from monai import transforms, data
import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 
from PIL import Image



from omegaconf import OmegaConf
import os
import torch
import cv2
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
from torchvision.transforms.functional import to_pil_image
from PIL import Image
os.chdir("/home/admin_mcn/thaotlp/Latent_DiffUnet")
from stablediffusion.ldm.util import instantiate_from_config
os.chdir("/home/admin_mcn/hungvq/stable_diffusion")


class ISICDataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist


    def read_data(self, data_path):
        filename = data_path.split("/")[-1]
        fileidx = filename.split(".")[0]
<<<<<<< HEAD
        mask_dir = '/home/admin_mcn/thaotlp/data/ISIC/mask'
        image_path = data_path
        mask_path = os.path.join(mask_dir, filename)

        image_data = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        mask_data = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

        image_data = np.transpose(np.array(image_data).astype(np.float32), (2,0,1))
        mask_data = np.expand_dims(np.array(mask_data).astype(np.int32), axis=0)
=======
        mask_dir = '/home/admin_mcn/minhtx/data/isic/mask'
        latent_dir = '/home/admin_mcn/hungvq/data/latent_gt'
        image_path = data_path
        mask_path = os.path.join(mask_dir, fileidx + '.jpg')

        latent_path = os.path.join(latent_dir, fileidx + '.npy')
        # image_data = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        image_data = Image.open(image_path)
        mask_data = Image.open(mask_path)
        latent_data = np.load(os.path.join(latent_path))[0]

        image_data = np.transpose(np.array(image_data).astype(np.float32), (2,0,1))
>>>>>>> 2d5202bb4e0b38ce8bf4b50e8560894e0271aafa
        return {
            "image": image_data,
            "mask": mask_data,
            "latent": latent_data,
            "name": fileidx
        } 

    def __getitem__(self, i):
<<<<<<< HEAD

        image = self.read_data(self.datalist[i])

        if self.transform is not None :
            image["image"] = self.transform(image["image"])
            image["mask"] = self.transform(image["mask"])

=======
        image = self.read_data(self.datalist[i])
        if self.transform is not None :
            image["image"] = self.transform(image["image"])
            mask_data = np.expand_dims(np.array(image["mask"]).astype(np.int32), axis=0)
            image["mask"] = self.transform(mask_data)
>>>>>>> 2d5202bb4e0b38ce8bf4b50e8560894e0271aafa
        return image

    def __len__(self):
        return len(self.datalist)




<<<<<<< HEAD
def get_loader_isic(data_dir, image_size = 512, batch_size=1, fold=0, num_workers=8):
=======
def get_loader_isic(data_dir, batch_size=1, fold=0, num_workers=8):
    image_size = 384
>>>>>>> 2d5202bb4e0b38ce8bf4b50e8560894e0271aafa

    all_dirs = os.listdir(data_dir)
    all_paths = [os.path.join(data_dir, d) for d in all_dirs]
   
    size = len(all_paths)
    train_size = int(0.7 * size)
    val_size = int(0.1 * size)
    train_files = all_paths[:train_size]
    val_files = all_paths[train_size:train_size + val_size]
    test_files = all_paths[train_size+val_size:]
    print(f"train is {len(train_files)}, val is {len(val_files)}, test is {len(test_files)}")

    transform_list = transforms.Compose(
<<<<<<< HEAD
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(),]
=======
        transforms.Resize((image_size, image_size)), transforms.ToTensor()
>>>>>>> 2d5202bb4e0b38ce8bf4b50e8560894e0271aafa
    )

    train_ds = ISICDataset(train_files, transform=transform_list)

    val_ds = ISICDataset(val_files, transform=transform_list)

    test_ds = ISICDataset(test_files, transform=transform_list)

    loader = [train_ds, val_ds, test_ds]

    return loader



if __name__ == '__main__':
    data_dir = '/home/admin_mcn/thaotlp/data/ISIC/image'
<<<<<<< HEAD
    batch_size = 2
    image_size = 512

    train_ds, val_ds, test_ds = get_loader_isic(data_dir)

    print(train_ds[0]["name"])
=======
    image_size = 384

    train_ds, val_ds, test_ds = get_loader_isic(data_dir)

    
    config_path = '/mnt/minhtx/VAE/2023-06-07T11-32-09-project.yaml'
    ckpt_path = '/mnt/minhtx/VAE/last.ckpt'

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])


    latent_gt = torch.Tensor(train_ds[100]["latent"])
    latent = model.decode(latent_gt)
    latent = latent[0].detach().cpu().numpy()

    
    image = np.transpose(train_ds[100]["image"].numpy(), (2,1,0))
    mask = np.transpose(train_ds[100]["mask"].numpy(), (2,1,0))

    # image = train_ds[100]["image"].numpy()
    # mask = train_ds[100]["mask"].numpy()
    print(image.shape)
    print(mask.shape)
    print(latent.shape)
    

    cv2.imwrite(os.path.join("/home/admin_mcn/thaotlp/Latent_DiffUnet/dataset", "image.png"), image) 
    cv2.imwrite(os.path.join("/home/admin_mcn/thaotlp/Latent_DiffUnet/dataset", "mask.png"), mask) 
    print(train_ds[0]["name"])

>>>>>>> 2d5202bb4e0b38ce8bf4b50e8560894e0271aafa
