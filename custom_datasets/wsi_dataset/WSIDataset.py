import abc

class WSIDataset(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod    
    def get_patches():
        pass
    
    @abc.abstractmethod        
    def get_WSIs():
        pass