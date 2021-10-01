
import yaml
from custom_datasets.LabelHandler import LabelHandler
from custom_datasets.patch.PatchFromFile import PatchFromFile
from pathlib import Path
from custom_datasets.patch.Patch import Patch
from typing import Dict, List, Union
from custom_datasets.wsi.WSI import WSI


class WSIFromFolder(WSI):
    
    def __init__(self,
                 root: str,
                 label_handler=None,
                 patch_wrapper=None):
        
        if label_handler is None:
            self.label_handler = LabelHandler()
        else:
            self.label_handler = label_handler
        
        self.root_path = Path(root)
        self.name = Path(root).stem
        self.patches = self.create_patches(root=self.root_path,
                                           patch_wrapper=patch_wrapper)
        self.meta_data = self.load_metadata()
    
    def load_metadata(self) -> dict[str, Union[str, int]]:
        
        config_path = self.root_path / "metadata.yaml"
        try:
            with config_path.open() as stream:
                    meta_data = yaml.safe_load(stream)
        except Exception as e:
            meta_data = dict()
        return meta_data
    
    def get_metadata(self, key: str):
        """Returns value of key's WSI metadata. If key does not exists, returns None
        """
        if key in self.meta_data.keys():
            return self.meta_data[key]
        else:
            None
    
    def create_patches(self, 
                       root, 
                       patch_wrapper=None
                       ) -> list[Patch]:
        
        if patch_wrapper is None:
            patch_wrapper = PatchFromFile
        
        patches = list()
        patches_files = [f for f in root.glob('**/*') if f.is_file() and f.suffix in ['.png', '.jpg']]
        for patch_file in patches_files:
            patches.append(patch_wrapper(file_path=patch_file, 
                                         label_handler=self.label_handler))
            
        return patches
    
    def get_patches(self) -> List[Patch]:
        return self.patches
    
    def get_label(self):
        pass
    