import abc
from typing import List
from custom_datasets.patch.Patch import Patch

class WSI(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def get_patches() -> List[Patch]:
        pass
    
    @abc.abstractmethod
    def get_label():
        pass
    