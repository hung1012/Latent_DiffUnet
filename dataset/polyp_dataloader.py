import os
import numpy as np
import torch
from monai import transforms, data
import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 
import cv2
from PIL import Image





class PolypDataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist


    def read_data(self, data_path):
        filename = data_path.split("/")[-1]
        fileidx = filename.split(".")[0]
        mask_dir = '/home/admin_mcn/minhtx/data/polyp/TrainDataset/masks'
        # latent_dir = '/home/admin_mcn/minhtx/data/polyp/TrainDataset/latent_gt'
        image_path = data_path
        mask_path = os.path.join(mask_dir, fileidx + '.png')
        # latent_path = os.path.join(latent_dir, fileidx + '.npy')

        # image_data = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        # mask_data = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        image_data = Image.open(image_path).convert('RGB')
        mask_data = Image.open(mask_path).convert('L')
        # latent_data = np.load(os.path.join(latent_path))[0]

        image_data = np.transpose(np.array(image_data).astype(np.float32), (2,0,1))
        # mask_data = np.transpose(np.array(mask_data).astype(np.float32), (2,0,1))
        return {
            "image": image_data,
            "mask": mask_data,
            "name": fileidx
        } 

    def __getitem__(self, i):
        image = self.read_data(self.datalist[i])
        if self.transform is not None :
            image["image"] = self.transform(image["image"])
            mask_data = np.expand_dims(np.array(image["mask"]).astype(np.int32), axis=0)
            image["mask"] = self.transform(mask_data)
            # image["mask"] = self.transform(image["mask"])
            
        return image

    def __len__(self):
        return len(self.datalist)




def get_loader_polyp(data_dir, batch_size=1, fold=0, num_workers=8):
    image_size = 384

    all_dirs = os.listdir(data_dir)
    all_paths = [os.path.join(data_dir, d) for d in all_dirs]
   
    size = len(all_paths)
    train_size = int(0.8 * size)
    val_size = int(0.2 * size)
    train_files = all_paths[:train_size]
    val_files = all_paths[train_size:train_size + val_size]
    # test_files = all_paths[train_size+val_size:]
    # print(f"train is {len(train_files)}, val is {len(val_files)}, test is {len(test_files)}")
    print(f"train is {len(train_files)}, val is {len(val_files)}")
    transform_list = transforms.Compose(
        transforms.Resize((image_size, image_size)), transforms.ToTensor()
    )

    train_ds = PolypDataset(train_files, transform=transform_list)

    val_ds = PolypDataset(val_files, transform=transform_list)

    # test_ds = PolypDataset(test_files, transform=transform_list)

    # loader = [train_ds, val_ds, test_ds]
    loader = [train_ds, val_ds]

    return loader


if __name__ == '__main__':
    data_dir = '/home/admin_mcn/minhtx/data/polyp/TrainDataset/images'
    image_size = 384

    train_ds, val_ds = get_loader_polyp(data_dir)

    image = np.transpose(train_ds[100]["image"].numpy(), (2,1,0))
    mask = np.transpose(train_ds[100]["mask"].numpy(), (2,1,0))

    # image = train_ds[100]["image"].numpy()
    # mask = train_ds[100]["mask"].numpy()
    # print(image.shape)
    # print(mask.shape)
    

    # cv2.imwrite(os.path.join("/home/admin_mcn/minhtx/Latent_DiffUnet/dataset", "image.png"), image) 
    # cv2.imwrite(os.path.join("/home/admin_mcn/minhtx/Latent_DiffUnet/dataset", "mask.png"), mask) 
    print(train_ds[0]["name"])
