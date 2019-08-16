def correr_VAE(data_name, num_run, z_dim, use_kldiv=True, sign_kldiv=1,
              mult_kldiv=1, epochs=5):
    """
    VAE para las esferas 3D
    Parametros:
        * data_name: nombre del dataset que hay que usar. El path es estandar
        dentro del repositorio.
        * num_run: numero de corrida. Sirve para ordenar los outputs de las
        distintas pruebas y mantener un orden.
        * z_dim: numero de parametros en el codigo latente.
    """
    import sys
    root_path = '/home/pablo/org/est/dm/mat/tesis_maestria_dm/repo/src/'
    sys.path.insert(0, root_path + 'vae_esferas/py_classes')
    sys.path.insert(0, root_path)

    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from skimage import io, transform

    import torch
    import torchvision
    from torchvision import datasets, transforms
    import torch.nn as nn
    import torch.functional as F

    import trimesh

    # Para importar datasets de esferas
    from HDF5SphDataset import HDF5SphDataset

    # Variational autoencoder class
    from VAE import VAE

    # Testeo si existe cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    learning_rate = 1e-3
    num_epochs = epochs

    RUTA_REPO = '/home/pablo/org/est/dm/mat/tesis_maestria_dm/repo/'
    data_path = RUTA_REPO + 'data/esferas/' + data_name

    def aTensor(sph):
        return {'sph': torch.Tensor(sph['sph'])}

    trf_composed = transforms.Compose([aTensor])

    train_sph_data = HDF5SphDataset(data_path, folder='train',
                                    transform=trf_composed)

    train_sph_data_loader = torch.utils.data.DataLoader(dataset=train_sph_data,
                                                       batch_size=4,
                                                       shuffle=True)

    # Modelando
    model = VAE(z_dim=z_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()
    criterionL1 = nn.SmoothL1Loss()
    criterionKLDiv = nn.KLDivLoss()

    z_codes = []
    # Start training
    model.train()
    for epoch in range(num_epochs):
        for i, x in enumerate(train_sph_data_loader):
            # Forward pass
            x = x['sph'].to(device)
            x_reconst, mu, log_var, z = model(x)

            if epoch == (num_epochs - 1) and len(z_codes) < 15:
                z_codes.append(z)

            # Compute reconstruction loss and kl divergence
            # For KL divergence, see Appendix B in VAE paper or
            # http://yunjey47.tistory.com/43

            # reconst_loss = F.binary_cross_entropy(x_reconst, x,
            # size_average=False)
            reconst_loss = criterion(x_reconst, x)
            reconst_L1 = criterionL1(x_reconst, x)

            # Esta kl div parece estar mal, uso la que tiene incorporada pytorch
            # kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            kl_div = criterionKLDiv(x_reconst, x)

            # Backprop and optimize
            if use_kldiv and sign_kldiv >= 0:
                loss = reconst_loss + reconst_L1 + mult_kldiv * kl_div
            elif use_kldiv and sign_kldiv < 0:
                loss = reconst_loss + reconst_L1 - mult_kldiv * kl_div
            elif use_kldiv is False:
                loss = reconst_loss + reconst_L1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Mu',mu.mean())
                print('Var',log_var.mean())
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                       .format(epoch+1, num_epochs, i+1, len(train_sph_data_loader), reconst_loss.item(), kl_div.item()))

    torch.save(model, RUTA_REPO + 'modelos/sph_vae_run_' + str(num_run) +
               '.pth')

    return z_codes

