"""
Class objects for standardization and validation of a model folder structure
"""

import os.path
from typing import Optional, List, Union
import numpy as np
import onnx
import yaml
import shutil



class File:
    """
    Object to wrap around common files. Currently, supporting:
    - numpy files
    - onnx files
    - yaml files
    - markdown

    """

    # TODO: If we are passing url, should we not download from it?
    def __init__(self, name: str, url: Optional[str] = None):
        self.name = name
        self.url = url
        self.supported_extensions = {'.npz': self._validate_numpy,
                                     '.onnx': self._validate_onnx,
                                     '.yaml': self._validate_yaml,
                                     '.md': self._validate_markdown,}

    # TODO: what is the purpose of `integration`
    def validate(self, integration: Optional[str] = None):
        _, extension = os.path.splitext(self.name)
        if extension not in self.supported_extensions.keys():
            raise ValueError(f"File {self.name} has an extension {extension}. The object {type(self).__name__} supports only following extensions: {list(self.supported_extensions.keys())}")
        validation_function = self.supported_extensions[extension]
        return validation_function()

    def _validate_numpy(self):
        if not np.load(self.name):
            raise ValueError("Numpy file could not been loaded properly")
        return True

    def _validate_onnx(self):
        if not onnx.load(self.name):
            raise ValueError("Onnx file could not been loaded properly")
        return True

    def _validate_yaml(self):
        try:
            with open(self.name) as file:
                yaml.load(file, Loader=yaml.FullLoader)
                return True
        except Exception as error:
            print("Yaml file could not been loaded properly")

    def _validate_markdown(self):
        pass


class Directory(File):
    @staticmethod
    def from_zip():
        files = None
        return files

    def __init__(self, files: Union[List[File]]):
        self.files = files
        self.dirname = self._validate()

    def _validate(self):
        """
        Validate the directory. This includes following steps:
        - make sure that all the files are located in one folder.
        - make sure that all files exist
        """
        dirnames = [os.path.dirname(file.name) for file in self.files]
        if len(set(dirnames)) != 1:
            raise ValueError("Not all of the passed files are located in the same directory.")
        for file in self.files:
            if not os.path.isfile(file.name):
                raise ValueError(f"Passed file {file.name} does not exists!")
        return dirnames[0]

    def gzip(self):
        return shutil.make_archive(self.dirname, 'zip', self.dirname)

    def _unzip(self):
        # TODO: Should this be implemented as `from_zip`?
        raise NotImplementedError()
    
class FrameworkFiles(Directory):
    def __init__(self, folders: List[Directory]):
        self.folders = folders
    def validate(self, integration: Optional[str]=None):
        pass

class SampleOriginals(Directory):
    def __init__(self, files: List[File]):
        super().__init__(files)
        self.extension = '.npz'

    def validate(self, integration: Optional[str]=None):
        for file in self.files:
            if not file._validate_numpy():
                _, extension = os.path.splitext(file.name)
                raise ValueError(f"File {file.name} has an extension {extension}. The object {type(self).__name__} supports only {self.extension} extensions.")
    # TODO: Are we building one iterator for all the numpy files?
    def iter_data(self):
        pass


class ModelDirectory(Directory):
    def __init__(self):
        pass

    def from_directory(self):
        pass

    def from_zoo_stub(self):
        pass

    def from_zoo_url(self):
        pass

    def from_zoo_api_json(self):
        pass

    def validate(self):
        pass

    def analyze(self):
        pass

    def generate_outputs(self, engine:str):
        pass

