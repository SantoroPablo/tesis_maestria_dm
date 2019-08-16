import sys
root_path = '/home/pablo/org/est/dm/mat/tesis_maestria_dm/repo/'
sys.path.insert(0, root_path + 'src/vae_esferas/py_classes')
sys.path.insert(0, root_path + 'src/')

import os
import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d

import torch
import torchvision
from torchvision import transforms

# Para importar los datasets de esferas
from HDF5SphDataset import HDF5SphDataset

from vae_sph import correr_VAE
from unet_sph import correr_unet
from VAE import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_name = 'esferas.hdf5'
data_path = root_path + 'data/esferas/' + data_name
model_path = root_path + 'modelos/sph_vae_run_7.pth'

model = VAE().to(device)
model = torch.load(model_path, map_location=lambda storage, loc: storage)
model.eval()

def aTensor(sph):
    return {'sph': torch.Tensor(sph['sph'])}

# Traigo el dataset de testing
trf_composed = transforms.Compose([aTensor])
test_sph_data = HDF5SphDataset(data_path, folder='testing',
                               transform=trf_composed)

test_sph_data_loader = torch.utils.data.DataLoader(dataset=test_sph_data,
                                                   batch_size=1,
                                                   shuffle=True)

testiter = iter(test_sph_data_loader)

sphereit = next(testiter)

list_orig = []
list_reconst = []
z_codes = []

with torch.no_grad():
    # Cantidad de esferas que voy a reconstruir para ver
    j = 10
    for i, x in enumerate(test_sph_data_loader):
        if i < j:
            reconst, _, _, z = model(x['sph'].to(device))
            z_codes.append(z)
            list_orig.append(x['sph'][0, 0, ...])
            list_reconst.append(reconst[0, 0, ...])

    # Guardo los archivos para luego armar grilla
    for i, x in enumerate(list_orig):
        fig = plt.figure(figsize=(20, 10))
        axes = fig.gca(projection='3d')
        axes.set_aspect('equal')
        x = x.to(device).numpy()
        axes.voxels(x, facecolors='y', edgecolors='k')
        fig.savefig(root_path + 'runs/comparison_vae_run_7/' +
                    'original_{}.png'.format(i),
                    bbox_inches='tight')
        plt.close()

    for i, x in enumerate(list_reconst):
        fig = plt.figure(figsize=(20, 10))
        axes = fig.gca(projection='3d')
        axes.set_aspect('equal')
        x = x.to(device).numpy()
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        mask = x > 0.5
        x[mask] = 1
        x[~mask] = 0
        axes.voxels(x, facecolors='y', edgecolors='k')
        fig.savefig(root_path + 'runs/comparison_vae_run_7/' +
                    'reconst_{}.png'.format(i),
                    bbox_inches='tight')
        plt.close()

orig_paths = glob.glob(root_path + 'runs/comparison_vae_run_7/original*')
orig_paths.sort()
reconst_paths = glob.glob(root_path + 'runs/comparison_vae_run_7/reconst*')
reconst_paths.sort()

list_orig = [cv2.imread(i) for i in orig_paths]
list_reconst = [cv2.imread(i) for i in reconst_paths]

nrows = 2
ncols = 10
plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 6))
gs = gridspec.GridSpec(nrows, ncols, wspace=0.0, hspace=0.0)

for i in range(nrows):
    for j in range(ncols):
        if i == 0:
            ax = plt.subplot(gs[i, j])
            ax.imshow(list_orig[j].squeeze())
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax = plt.subplot(gs[i,j])
            ax.imshow(list_reconst[j].squeeze())
            ax.set_xticklabels([])
            ax.set_yticklabels([])

plt.show()

# Analisis de los parametros del codigo latente
# Por lo que veo de los z-codes, parece que tienen una variabilidad grande
# Voy a probar a moverlos en el rango [-100; 100]
z_codes = np.random.rand(*(100, 4)) * 200 - 100

# muevo los primeros dos parametros y dejo fijo los ultimos dos en cero
p1 = np.linspace(-100, 100, 11)
p2 = np.linspace(-100, 100, 11)
p3 = 0
p4 = 0

with torch.no_grad():
    for i in p1:
        for j in p2:
            z_code = torch.Tensor(np.array([i, j, p3, p4])).to('cpu')
            x = model.decode(z_code).numpy()[0, 0, ...]
            fig = plt.figure(figsize=(20, 10))
            axes = fig.gca(projection='3d')
            axes.set_aspect('equal')
            mask = x > 0.5
            x[mask] = 1
            x[~mask] = 0
            axes.voxels(x, facecolors='y', edgecolors='k')
            fig.savefig(root_path + 'runs/ceteris_paribus/' +
                        'reconst_{}_{}_{}_{}.png'.format(i, j, p3, p4),
                        bbox_inches='tight')
            plt.close()

    nrows = 11
    ncols = 11

    # Tomando las imagenes para mostrar en forma de matriz
    sph_path = root_path + 'runs/ceteris_paribus'
    sph_path = glob.glob(sph_path + '/*')
    list_reconst = [cv2.imread(i) for i in sph_path]

    plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 6))
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.0, hspace=0.0)
    for i_num, i in enumerate(list_reconst):
        ax = plt.subplot(11, 11, i_num + 1)
        ax.imshow(i.squeeze())

# parece que estos dos parametros son de tamanio

# Armo matriz para visualizar 


# TODO: armar una matriz con z_codes yendo entre [-30; 30] y [-20; 20]

