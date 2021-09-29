from typing import Union
from custom_datasets.wsi.WSI import WSI
import torch
from torchvision.transforms.functional import pil_to_tensor

from custom_datasets.wsi_dataset.WSIDataset import WSIDataset
from torch.utils.data import Dataset

class CustomPatchesDataset(Dataset):
    
    """WSI patches dataset."""

    def __init__(self, wsi_dataset: Union[WSIDataset, WSI], transform=None):
        """[summary]

        Args:
            wsi_dataset (WSIDataset): [description]
            transform ([type], optional): [description]. Defaults to None.
        """
        
        self.wsi_dataset = wsi_dataset
        self.patches = self.wsi_dataset.get_patches()
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patch_img, label = self.patches[idx]()
        
        if self.transform is not None:
            patch_img = self.transform(patch_img)
            
        #patch = pil_to_tensor(patch_img)
        return patch_img, label