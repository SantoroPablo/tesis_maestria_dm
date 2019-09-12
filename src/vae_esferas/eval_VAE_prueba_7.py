"""
Evaluacion de la prueba numero 7, para visualizar la reconstruccion de las
esferas hechas por el modelo
"""
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d

import glob
import cv2
import os
import sys
import random

random.seed(0)

root_path = '/home/pablo/org/est/dm/mat/tesis_maestria_dm/repo/'
sys.path.insert(0, root_path + 'src/vae_esferas/py_classes')
sys.path.insert(0, root_path + 'src/vae_esferas/')
sys.path.insert(0, root_path + 'src/')

from HDF5SphDataset import HDF5SphDataset
from VAE import VAE

model = VAE()

root_path = '/home/pablo/org/est/dm/mat/tesis_maestria_dm/repo/'
data_path = root_path + 'data/esferas/esferas.hdf5'
model_path = root_path + 'modelos/sph_vae_run_12.pth'

model = torch.load(model_path, map_location=lambda storage, loc:storage)
model.eval()

def aTensor(sph):
    return {'sph': torch.Tensor(sph['sph'])}

# Datos de testeo
trf_composed = transforms.Compose([aTensor])
test_sph_data = HDF5SphDataset(data_path, folder='testing',
                               transform = trf_composed)
test_sph_data_loader = torch.utils.data.DataLoader(dataset=test_sph_data,
                                                   batch_size=1,
                                                   shuffle=True)

z_codes = [] # Voy a guardar los 100 codigos latentes generados en testing
with torch.no_grad():
    for i, x in enumerate(test_sph_data_loader):
        _, _, _, z = model(x['sph'])
        z_codes.append(z.numpy())

z_codes = np.stack(z_codes)[:, 0, :]

maxim = z_codes.max(axis=0)
minim = z_codes.min(axis=0)

# p1 y p2 van a hacer los dos primeros parametros que voy a mover.
# Los otros dos los voy a dejar fijos en su media.

p1_sp = np.linspace(minim[0], maxim[0], 10)
p2_sp = np.linspace(minim[1], maxim[1], 10)
p3 = z_codes.mean(axis=0)[2]
p4 = z_codes.mean(axis=0)[3]

prueba = torch.Tensor(np.array([p1_sp[0], p2_sp[0], p3, p4]))
x = model.decode(prueba).detach().numpy()[0, 0, ...]
# x = (prueba - prueba.min()) / (prueba.max() - prueba.min())
fig = plt.figure(figsize=(20, 10))
axes = fig.gca(projection='3d')
axes.set_aspect('equal')
mask = x > 0.5
x[mask] = 1
x[~mask] = 0
axes.voxels(x, facecolors='y', edgecolors='k')
plt.savefig(root_path + 'salida.jpg')

# Armar la matriz de esferas para ver como se reconstruyen
random.seed(0)
image_list = []
with torch.no_grad():
    for i in p1_sp:
        for j in p2_sp:
            z_code = torch.Tensor(np.array([i, j, p3, p4])).to('cpu')
            x = model.decode(z_code).numpy()[0, 0, ...]
            fig = plt.figure(figsize=(20, 10))
            axes = fig.gca(projection='3d')
            axes.set_aspect('equal')
            # x = (x - x.min()) / (x.max() - x.min())
            mask = x > 0.5
            x[mask] = 1
            x[~mask] = 0
            axes.voxels(x, facecolors='y', edgecolors='k')
            figname = root_path + 'runs/ceteris_paribus/p1_p2/'
            figname += 'reconst_{:.2f}_{:.2f}_{:.2f}_{:.2f}.png'.format(i,
                    j, p3, p4)
            fig.savefig(figname, bbox_inches='tight')
            image_list.append(cv2.imread(figname))
            plt.close()

nrows = 10
ncols = 10

# Tomando las imagenes para mostrar en forma de matriz
plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 6))
gs = gridspec.GridSpec(nrows, ncols, wspace=0.0, hspace=0.0)
for i_num, i in enumerate(image_list):
    ax = plt.subplot(nrows, ncols, i_num + 1)
    ax.imshow(i.squeeze())

plt.savefig(root_path + 'salida.jpg')

# TODO: armar matplotlib interactivo para poder revisar los cambios de
#       parametros. En 3D. Si es posible, armar gif o video de ello.
from matplotlib.widgets import Slider

np.random.seed(0)
p1 = np.random.uniform(minim[0], maxim[0])
p2 = np.random.uniform(minim[1], maxim[1])
p3 = np.random.uniform(minim[2], maxim[2])
p4 = np.random.uniform(minim[3], maxim[3])

x = model.decode(torch.Tensor([p1, p2, p3, p4]))[0, 0, ...]
x = x.detach().numpy()
fig = plt.figure(figsize=(20, 10))
axes = fig.gca(projection='3d')
axes.set_aspect('equal')
mask = x > 0.5
x[mask] = 1
x[~mask] = 0
axes.voxels(x, facecolors='y', edgecolors='k')

# Axes
axcolor = 'lightgoldenrodyellow'
axp1 = fig.add_axes([0.1, 0.1, 0.2, 0.03], facecolor=axcolor)
axp2 = fig.add_axes([0.1, 0.15, 0.2, 0.03], facecolor=axcolor)
axp3 = fig.add_axes([0.1, 0.2, 0.2, 0.03], facecolor=axcolor)
axp4 = fig.add_axes([0.1, 0.25, 0.2, 0.03], facecolor=axcolor)

# Sliders
slp1 = Slider(axp1, 'p1', float(minim[0]), float(maxim[0]), valinit=p1)
slp2 = Slider(axp2, 'p2', float(minim[1]), float(maxim[1]), valinit=p2)
slp3 = Slider(axp3, 'p3', float(minim[2]), float(maxim[2]), valinit=p3)
slp4 = Slider(axp4, 'p4', float(minim[3]), float(maxim[3]), valinit=p4)

def update(val):
    p1 = slp1.val
    p2 = slp2.val
    p3 = slp3.val
    p4 = slp4.val
    axes.clear()
    x = model.decode(torch.Tensor([p1, p2, p3, p4]))[0, 0, ...]
    x = x.detach().numpy()
    mask = x > 0.5
    x[mask] = 1
    x[~mask] = 0
    axes.voxels(x, facecolors='y', edgecolors='k')
    fig.canvas.draw_idle()

slp1.on_changed(update)
slp2.on_changed(update)
slp3.on_changed(update)
slp4.on_changed(update)
plt.show()

# 03/09/2019
# Las esferas no estan saliendo bien. Voy a probar a ver si reconstruye
# bien las esferas de testing
test_iter = iter(test_sph_data_loader)
test = next(test_iter)
test = test['sph']

with torch.no_grad():
    result = model(test)
    x = model.decode(result[1])[0, 0, ...].numpy()
    fig = plt.figure(figsize=(20, 10))
    axes = fig.gca(projection='3d')
    axes.set_aspect('equal')
    mask = x > 0.5
    x[mask] = 1
    x[~mask] = 0
    axes.voxels(x, facecolors='y', edgecolors='k')
    plt.savefig(root_path + 'salida.jpg')

