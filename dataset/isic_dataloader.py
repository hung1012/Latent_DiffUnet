import os
import json
import math
import numpy as np
import torch
from torchvision.transforms import Resize, ToTensor, Compose
from PIL import Image
import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 
# import pandas as pd


class ISICDataset(Dataset):
    def __init__(self, datalist, masklist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.masklist = masklist


    def read_data(self, data_path, mask_path):
        filename = data_path.split("/")[-1]
        fileidx = filename.split(".")[0]
        latent_dir = '/home/admin_mcn/hungvq/data/latent_gt'
        image_path = data_path
        latent_path = os.path.join(latent_dir, fileidx + '.npy')

        # image_data = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        # mask_data = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        image_data = Image.open(image_path)
        mask_data = Image.open(mask_path)
        latent_data = np.load(os.path.join(latent_path))[0]

        # image_data = np.transpose(np.array(image_data).astype(np.float32), (2,0,1))
        # mask_data = np.array(mask_data).astype(np.float32)

        return {
            "image": image_data,
            "latent": latent_data,
            "mask": mask_data,
            "name": fileidx
        } 

    def __getitem__(self, i):
        image = self.read_data(self.datalist[i], self.masklist[i])
        if self.transform is not None :
            image["image"] = self.transform(image["image"])
            image["mask"] = self.transform(image["mask"])
        return image

    def __len__(self):
        return len(self.datalist)




def get_loader_isic(data_dir, mask_dir, batch_size=1, fold=0, num_workers=8):
    image_size = 384

    all_dirs = os.listdir(data_dir)
    all_paths = [os.path.join(data_dir, d) for d in all_dirs]
    
    all_dirs_mask = os.listdir(mask_dir)
    all_paths_mask = [os.path.join(mask_dir, d) for d in all_dirs_mask]
   
    size = len(all_paths)
    train_size = int(0.7 * size)
    val_size = int(0.1 * size)
    train_files = all_paths[:train_size]
    train_mask = all_paths_mask[:train_size]
    val_files = all_paths[train_size:train_size + val_size]
    val_mask = all_paths_mask[:train_size:train_size + val_size]
    test_files = all_paths[train_size+val_size:]
    test_mask = all_paths_mask[train_size+val_size:]
    print(f"train is {len(train_files)}, val is {len(val_files)}, test is {len(test_files)}")

    transform_list = Compose([
        Resize((image_size, image_size)), ToTensor()]
    )

    train_ds = ISICDataset(train_files, train_mask, transform=transform_list)

    val_ds = ISICDataset(val_files, val_mask, transform=transform_list)

    test_ds = ISICDataset(test_files, test_mask, transform=transform_list)

    loader = [train_ds, val_ds, test_ds]

    return loader


if __name__ == '__main__':
    data_dir = '/home/admin_mcn/thaotlp/data/ISIC/image'
    mask_dir = '/home/admin_mcn/thaotlp/data/ISIC/mask'
    batch_size = 1
    image_size = 512

    train_ds, val_ds, test_ds = get_loader_isic(data_dir, mask_dir)

    print(test_ds[0]["name"])
