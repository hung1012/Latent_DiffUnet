import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random


class PolypMaskBase(Dataset):
    def __init__(self,
                 size=384,
                 interpolation="bicubic",
                 flip_p=0.5,
                 flag=None
                 ):
        self.data_root = '/home/admin_mcn/hungvq/out_data'
        self.list_path = (os.listdir(self.data_root))
        list_mask = [k for k in self.list_path if 'mask' in k]
        self.list_mask = random.shuffle(list_mask)
        self.flag = flag
        split_size = len(list_mask) * 8 // 10
        if flag=="train":
            self.list_mask = list_mask[:split_size]
        elif flag=="val":
            self.list_mask = list_mask[split_size:]
        self._length = len(self.list_mask)
        # self.list_mask = os.listdir(self.data_root)
        self.labels = {
            "relative_file_path_": "test",
            "file_path_": "test",
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k]) for k in self.labels)
        # image = Image.open(example["file_path_"])
        image = Image.open(os.path.join(self.data_root, self.list_mask[i]))
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example

class PolypTrain(PolypMaskBase):
    def __init__(self, size=384, interpolation="bicubic", flip_p=0.5, flag="train"):
        super().__init__(size, interpolation, flip_p, flag)

class PolypValidate(PolypMaskBase):
    def __init__(self, size=384, interpolation="bicubic", flip_p=0.5, flag="val"):
        super().__init__(size, interpolation, flip_p, flag)
