from pathlib import Path
from custom_datasets.LabelHandler import LabelHandler
from custom_datasets.patch.Patch import Patch
from PIL import Image

class PatchFromFile(Patch):
    
    def __init__(self, 
                 file_path: str,
                 label_handler: LabelHandler) -> None:
        
        self.file_path = file_path
        self.x, self.y, self.org_label = self._parse_file(self.file_path)
        self.label = label_handler.encode(self.org_label)
        
    def _parse_file(self, file_path: str):
        basename = Path(file_path).stem
        name_parts = basename.split('_')
        x = int(name_parts[-2])
        y = int(name_parts[-3])
        label = name_parts[-1]
        
        return x, y, label
        
    def get_coordinates(self):
        return self.x, self.y
    
    def __call__(self):
        image = Image.open(self.file_path)
        return image, self.label

