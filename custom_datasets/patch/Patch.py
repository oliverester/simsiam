import abc

class Patch(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def __call__(self):
        pass
    