# Clase para cargar el dataset de esferas desde un dataset HDF5
# Es una subclase de la clase 'Dataset' de pytorch
import torch.utils.data
import h5py
import os
import numpy as np

class HDF5SphDataset(torch.utils.data.Dataset):
    """
    Dataset de esferas 3D
    """
    def __init__(self, file_path, folder, data_cache_size=3, transform=None):
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform
        self.folder = folder
        assert os.path.exists(file_path)
        self.file_path = file_path
        with h5py.File(file_path, 'r') as f:
            self.nombres = [name for name in f[folder]]

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            return len(f[self.folder])

    def __getitem__(self, idx):
        with h5py.File(self.file_path) as f:
            matriz = f[self.folder][self.nombres[idx]][:]
        sample = {'sph': matriz[np.newaxis, ...]} # Le agrego un canal
        if self.transform:
            sample = self.transform(sample)
        return sample
