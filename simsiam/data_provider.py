from typing import List
from custom_datasets.patch.PatchFromFile import PatchFromFile
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import simsiam
import simsiam.loader
from ship_ai.pre_histo.pytorch_datasets.CustomPatchesDataset import CustomPatchesDataset
from ship_ai.pre_histo.pytorch_datasets.wsi_dataset.WSIDatasetFolder import WSIDatasetFolder


class DataProvider():
    
    def __init__(self, args):
        
        self.args = args
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        
        self.non_aug_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize
        ]
        
        self.aug_transform = simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation))

    def get_train_loader(self, aug=True, sampling='up'):
        # Data loading code
        traindir = self.args.train_data
        
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        
        #train_dataset = datasets.ImageFolder(
        #     traindir,
        #     self.aug_transform if aug else self.non_aug_transform)

        train_dataset = CustomPatchesDataset(wsi_dataset=WSIDatasetFolder(root_folder=traindir, 
                root_contains_wsi_label=False, sampling=sampling, debug=False),
                transform=self.aug_transform if aug else self.non_aug_transform)
        
        if 'distributed' in self.args and self.args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        print(f"GPU {self.args.gpu} Train Data set length {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=int(self.args.batch_size), shuffle=(train_sampler is None),
            num_workers=self.args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
        
        return train_loader, train_sampler
    
    def get_val_loader(self, sampling=None):
        
        validir = self.args.test_data
        
        #val_dataset = datasets.ImageFolder(validir, self.non_aug_transform)

        val_dataset = CustomPatchesDataset(wsi_dataset=WSIDatasetFolder(root_folder=validir, root_contains_wsi_label=False, sampling=sampling, debug=False), 
                                           transform=self.non_aug_transform)
        # validation loader does not have distributed loader. So, each GPU runs a full validation run. However, only rank 0 prints
        val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256, shuffle=False,
        num_workers=self.args.workers, pin_memory=True)
    
        return val_loader
    
    def get_wsi_loader(self, wsi) -> torch.utils.data.DataLoader:
        """Generates a DataLoader from a WSI object (generated by pre-histo)

        Returns:
            torch.utils.data.DataLoader: DataLoader
        """
        wsi_dataset = CustomPatchesDataset(wsi_dataset=wsi, 
                                           transform=self.non_aug_transform)
        # validation loader does not have distributed loader. So, each GPU runs a full validation run. However, only rank 0 prints
        wsi_loader = torch.utils.data.DataLoader(wsi_dataset,
                                                 batch_size=256, shuffle=False,
                                                 num_workers=self.args.workers, pin_memory=True)
                    
        return wsi_loader

