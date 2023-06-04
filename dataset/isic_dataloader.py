import os
import json
import math
import numpy as np
import torch
from monai import transforms, data
import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 
# import pandas as pd


class ISICDataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist


    def read_data(self, data_path):
        filename = data_path.split("/")[-1]
        fileidx = filename.split(".")[0]
        latent_dir = '/home/admin_mcn/thaotlp/data/ISIC/latent_gt'
        image_path = data_path
        latent_path = os.path.join(latent_dir, fileidx + '.npy')

        image_data = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        latent_data = np.load(os.path.join(latent_path))[0]

        image_data = np.transpose(np.array(image_data).astype(np.float32), (2,0,1))
        return {
            "image": image_data,
            "latent": latent_data,
            "name": fileidx
        } 

    def __getitem__(self, i):
        image = self.read_data(self.datalist[i])
        if self.transform is not None :
            image["image"] = self.transform(image["image"])
        return image

    def __len__(self):
        return len(self.datalist)




def get_loader_isic(data_dir, batch_size=1, fold=0, num_workers=8):
    image_size = 384

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
        transforms.Resize((image_size, image_size)), transforms.ToTensor()
    )

    train_ds = ISICDataset(train_files, transform=transform_list)

    val_ds = ISICDataset(val_files, transform=transform_list)

    test_ds = ISICDataset(test_files, transform=transform_list)

    loader = [train_ds, val_ds, test_ds]

    return loader


if __name__ == '__main__':
    data_dir = '/home/admin_mcn/thaotlp/data/ISIC/image'
    batch_size = 2
    image_size = 512

    train_ds, val_ds, test_ds = get_loader_isic(data_dir)

    print(test_ds[0]["name"])
