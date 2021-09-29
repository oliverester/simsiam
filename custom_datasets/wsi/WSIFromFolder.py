
from custom_datasets.patch.PatchFromFile import PatchFromFile
from pathlib import Path
from custom_datasets.patch.Patch import Patch
from typing import Dict, List, Union
from custom_datasets.wsi.WSI import WSI


class WSIFromFolder(WSI):
    
    def __init__(self,
                 root: str,
                 patch_wrapper=None):
        
        self.root_path = Path(root)
        self.patches = self.create_patches(root=self.root_path,
                                           patch_wrapper=patch_wrapper)
    
    def get_metadata(self) -> dict[str, Union[str, int]]:
        pass
    
    def create_patches(self, 
                       root, 
                       patch_wrapper=None
                       ) -> list[Patch]:
        
        if patch_wrapper is None:
            patch_wrapper = PatchFromFile
        
        patches = list()
        patches_files = [f for f in root.glob('**/*') if f.is_file() and f.suffix in ['.png', '.jpg']]
        for patch_file in patches_files:
            patches.append(patch_wrapper(file_path=patch_file))
            
        return patches
    
    def get_patches(self) -> List[Patch]:
        return self.patches
    
    def get_label(self):
        pass
    