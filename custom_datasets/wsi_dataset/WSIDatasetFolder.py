
from pathlib import Path
from custom_datasets.LabelHandler import LabelHandler
from custom_datasets.wsi.WSI import WSI
from custom_datasets.wsi.WSIFromFolder import WSIFromFolder

from typing import List
from custom_datasets.patch.Patch import Patch

class WSIDatasetFolder():
    
    def __init__(self,
                 root_folder: str,
                 wsi_wrapper=None,
                 patch_wrapper=None
                 ) -> None:
        
        self.root_path = Path(root_folder)
        
        # instance to manage labels
        self.label_handler = LabelHandler()
        
        self.wsi_list = self.create_WSIs(root=self.root_path,
                                         wsi_wrapper=wsi_wrapper,
                                         patch_wrapper=patch_wrapper,
                                         label_handler=self.label_handler)
        self.patches = self.collect_patches(self.wsi_list)
    
    def collect_patches(self, 
                        wsi_list: List[WSI]
                        ) -> List[Patch]:
        patch_list = list()
        for wsi in wsi_list:
            patch_list.extend(wsi.get_patches())
        return patch_list
    
    def create_WSIs(self, 
                    root: Path,
                    patch_wrapper,
                    wsi_wrapper,
                    label_handler
                    ) -> List[WSI]:
        
        if not root.exists():
            raise Exception(f"WSI dataset folder {root} does not exists")
        
        if wsi_wrapper is None:
            wsi_wrapper = WSIFromFolder
        
        wsi_list = list()
        
        wsi_roots = [d for d in root.iterdir() if d.is_dir()]
        for wsi_root in wsi_roots:
            print(f"Creating WSI object for {wsi_root.name}")
            wsi = wsi_wrapper(root=wsi_root, 
                              patch_wrapper=patch_wrapper,
                              label_handler=label_handler)
            wsi_list.append(wsi)
        
        return wsi_list
        
    def get_WSIs(self) -> List[WSI]:
        return self.wsi_list
            
    def get_patches(self) -> List[Patch]:
        return self.patches
    