# Clase para cargar el dataset de esferas a partir de
# la lectura de una lista de archivos en un CSV.
# Es una subclase de la clase 'Dataset' de pytorch
import torch
import torchvision
import numpy as np
import pandas as pd
import trimesh
import os

class CSVSphDataset(torch.utils.data.Dataset):
    """
    Dataset de esferas 3D
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.sph_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.sph_frame)

    def __getitem__(self, idx):
        sph_name = os.path.join(self.root_dir, self.sph_frame.iloc[idx, 0])
        sph = trimesh.load(sph_name)
        sample = {'sph': sph}

        if self.transform:
            sample = self.transform(sample)

        return sample
