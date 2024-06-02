from torch.utils import data
from torchvision import transforms
import numpy as np


class vessel_dataset(data.Dataset):
    def __init__(self, patches_imgs, patches_masks, split_ratio=1.0, split="train"):
        assert split in ["train", "test", "val"]
        self.split = split
        shuffle_indices = np.arange(len(patches_imgs))
        np.random.shuffle(shuffle_indices)
        split_index = int(split_ratio * len(patches_imgs))
        if self.split == "train":
            self.imgs = patches_imgs[shuffle_indices][:split_index]
            self.masks = patches_masks[shuffle_indices][:split_index]
        elif self.split == "val":
            self.imgs = patches_imgs[shuffle_indices][split_index:]
            self.masks = patches_masks[shuffle_indices][split_index:]
        else:
            self.imgs = patches_imgs[shuffle_indices][:split_index]
        self.img_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.split == "train" or self.split == "val":
            return self.img_transform(self.imgs[index]), self.img_transform(self.masks[index])
        else:
            return self.img_transform(self.imgs[index])
