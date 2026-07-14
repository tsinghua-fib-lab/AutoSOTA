from abc import ABC, abstractmethod

class Template(ABC):
    verbose = False

    def __init__(self):
        ...

    def __repr__(self):
        ... 

    @abstractmethod
    def is_match(self, event):
        ...
    
    @abstractmethod
    def load_templates(self, template_file):
        ...
