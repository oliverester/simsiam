from pathlib import Path
from custom_datasets.patch.Patch import Patch
from PIL import Image

class PatchFromFile(Patch):
    
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        
        self.x, self.y, self.label = self._parse_file(self.file_path)
        
    def _parse_file(self, file_path: str):
        basename = Path(file_path).stem
        name_parts = basename.split('_')
        x = name_parts[-3]
        y = name_parts[-2]
        label = name_parts[-1]
        
        return x, y, label
        
    def get_coordinates(self):
        return self.x, self.y
    
    def __call__(self):
        image = Image.open(self.file_path)
        return image, self.label

