import torch
import os
import tempfile
import copy

class StateCacher:
    def __init__(self, in_memory:bool =False, path:str=None, overwrite:bool = False):

        self.in_memory = in_memory

        if in_memory:
            self.memmory_cache = None
            self.delete=False
        else:
            if path is None:
                self.path = tempfile.gettempdir()+'temp.ckpt'
                self.delete = True
            else:
                self.path = path
                if (not overwrite) and (not os.path.isfile(path)):
                    raise ValueError(f"File {path} already exists.")
                self.delete = False
        
        self.empty = True

    def save(self, module):
        if self.in_memory:
            self.memmory_cache = copy.deepcopy(module.state_dict())
        else:
            torch.save(module.state_dict(), self.path)
        self.empty = False

    def load(self, module):
        if self.in_memory:
            module.load_state_dict(self.memmory_cache)
        else:
            module.load_state_dict(torch.load(self.path))

    def delete_temp(self):
        r"""Deletes temporary files"""
        if self.delete and os.path.exists(self.path):
            os.remove(self.path)
        self.empty = True

    def __del__(self):
        r"""Checks if there is a file to delete"""
        self.delete_temp()

        