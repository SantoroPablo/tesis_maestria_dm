def correr_pruebas_testing(modelo_class, modelo_path, data_name, prefix,
                           num_run):
    """
    Script para correr pruebas sobre los modelos generados de VAE sobre el
    dataset de esferas.

    Parametros:
        * modelo_class: la clase del modelo usada para reconstruir la esfera
        * modelo_path: nombre del modelo que se va a cargar para hacer la
        prueba. El directorio desde donde se lo va a cargar es estandar en el
        repositorio.
        * data_name: nombre del dataset a usar. El directorio desde donde se lo
        va a cargar es estandar.
        * prefix: prefijo para los archivos de salida del analisis.
        * num_run: numero de corrida, es estandar y sirve para ordenar la salida
        de distintas pruebas.
    """
    import sys
    root_path = '/home/pablo/org/est/dm/mat/tesis_maestria_dm/repo/'
    sys.path.insert(0, root_path + 'src/vae_esferas/py_classes')
    sys.path.insert(0, root_path + 'src/')

    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    from skimage import io, transform

    import torch
    import torchvision
    from torchvision import datasets, transforms
    import torch.nn as nn
    import torch.functional as F

    # Para importar datasets de esferas
    from HDF5SphDataset import HDF5SphDataset
    from scipy import stats

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = root_path + 'data/esferas/' + data_name
    model_path = root_path + 'modelos/' + modelo_path

    model = modelo_class.to(device)
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()

    def aTensor(sph):
        return {'sph': torch.Tensor(sph['sph'])}

    # Traigo el dataset de testing
    trf_composed = transforms.Compose([aTensor])
    test_sph_data = HDF5SphDataset(data_path, folder='testing',
                                   transform=trf_composed)

    test_sph_data_loader = torch.utils.data.DataLoader(dataset=test_sph_data,
                                                       batch_size=4,
                                                       shuffle=True)

    testiter = iter(test_sph_data_loader)

    sphereit = next(testiter)
    sphere = sphereit['sph']

    np_sphere = sphere[1, :, :, :].numpy()
    np_sphere = np_sphere[0, :, :, :]

    print('original')
    fig = plt.figure(figsize=(20, 10))
    axes = fig.gca(projection='3d')
    axes.set_aspect('equal')
    axes.voxels(np_sphere, facecolors='y', edgecolors='k')
    fig.savefig(root_path + 'runs/' + prefix + '_cube_sph_orig_run_' +
                str(num_run) + '.png')
    plt.close()

    print('reconstruccion')
    with torch.no_grad():
        np_reconst_sph, mu, logvar, z = model(sphereit['sph'].to(device))
        np_reconst_sph = np_reconst_sph[1, :, :, :].numpy()
        np_reconst_sph = np_reconst_sph[0, :, :, :]
        np_reconst_sph_orig = np_reconst_sph
        np_reconst_sph = (np_reconst_sph - np.min(np_reconst_sph)) / (
            np.max(np_reconst_sph) - np.min(np_reconst_sph)
        )
        # tengo que usar una mascara porque sino los valores casi cero los plotea
        mask = np_reconst_sph > 0.5
        np_reconst_sph[mask] = 1
        np_reconst_sph[~mask] = 0
        fig = plt.figure(figsize=(20, 10))
        axes = fig.gca(projection='3d')
        axes.set_aspect('equal')
        axes.voxels(np_reconst_sph, facecolors='y', edgecolors='k')
        fig.savefig(root_path + 'runs/' + prefix + '_cube_sph_reconst_run_' +
                    str(num_run) + '.png')
        plt.close()

    # histograma de los valores de la reconstruccion
    num_bins = 30
    fig = plt.figure()
    fig = plt.hist(np_reconst_sph_orig.flatten(), num_bins, facecolor='blue',
                   alpha=0.5)
    plt.savefig(root_path + 'runs/' + prefix + '_hist_reconst_run_' +
                str(num_run) + '.png')
    plt.close()

    return z

