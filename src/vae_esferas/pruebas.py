import sys
root_path = '/home/pablo/org/est/dm/mat/tesis_maestria_dm/repo/'
sys.path.insert(0, root_path + 'src/vae_esferas/py_classes')
sys.path.insert(0, root_path + 'src/vae_esferas/')
sys.path.insert(0, root_path + 'src/')

from eval_runs import correr_pruebas_testing
from vae_sph import correr_VAE
from unet_sph import correr_unet
from VAE import VAE
from VAE_direct_mu_sigma import VAE_direct
from unet import MyUNet

vae_model = VAE
vae_dir_model = VAE_direct
unet_model = MyUNet()

# Primera prueba
num_run = 1
z_dim = 4
correr_VAE(vae_model, 'esferas.hdf5', num_run, z_dim)
z_codes = correr_pruebas_testing(vae_model, 'sph_vae_run_' + str(num_run) +
                                 '.pth', 'esferas.hdf5', 'vae', num_run)

# Segunda prueba
num_run = 2
z_dim = 4
correr_VAE(vae_model, 'esferas.hdf5', num_run, z_dim, sign_kldiv=-1)
z_codes = correr_pruebas_testing(vae_model, 'sph_vae_run_' + str(num_run) +
                                 '.pth', 'esferas.hdf5', 'vae', num_run)

# Tercera prueba
num_run = 3
z_dim = 4
correr_VAE(vae_model, 'esferas.hdf5', num_run, z_dim, use_kldiv=False)
z_codes = correr_pruebas_testing(vae_model, 'sph_vae_run_' + str(num_run) +
                                 '.pth', 'esferas.hdf5', 'vae', num_run)

# Cuarta prueba
num_run = 4
z_dim = 10
correr_VAE(vae_model, 'esferas.hdf5', num_run, z_dim)
z_codes = correr_pruebas_testing(vae_model, 'sph_vae_run_' + str(num_run) +
                                 '.pth', 'esferas.hdf5', 'vae', num_run)

# Quinta prueba
num_run = 5
z_dim = 15
correr_VAE(vae_model, 'esferas.hdf5', num_run, z_dim)
z_codes = correr_pruebas_testing(vae_model, 'sph_vae_run_' + str(num_run) +
                                 '.pth', 'esferas.hdf5', 'vae', num_run)

# Sexta prueba
num_run = 6
z_dim = 20
correr_VAE(vae_model, 'esferas.hdf5', num_run, z_dim)
z_codes = correr_pruebas_testing(vae_model, 'sph_vae_run_' + str(num_run) +
                                 '.pth', 'esferas.hdf5', 'vae', num_run)

# Septima prueba
num_run = 7
z_dim = 4
num_epochs = 15
correr_VAE(vae_model, 'esferas.hdf5', num_run, z_dim, epochs=num_epochs)
z_codes = correr_pruebas_testing(vae_model, 'sph_vae_run_' + str(num_run) +
                                 '.pth', 'esferas.hdf5', 'vae', num_run)

# Octava prueba
num_run = 8
z_dim = 1000
num_epochs = 20
correr_VAE(vae_model, 'esferas.hdf5', num_run, z_dim, epochs=num_epochs)

# Novena prueba
num_run = 9
z_dim = 4
num_epochs = 10
correr_VAE(vae_model, 'esferas.hdf5', num_run, z_dim, use_kldiv=False, use_bce=False)

# Novena prueba
num_run = 10
z_dim = 4
num_epochs = 20
correr_VAE(vae_model, 'esferas.hdf5', num_run, z_dim, mult_kldiv=100, epochs=num_epochs)

# Onceava prueba
num_run = 11
z_dim = 4
num_epochs = 20
correr_VAE(vae_model, 'esferas.hdf5', num_run, z_dim, mult_kldiv=1000, epochs=num_epochs)

# Doceava prueba
# Prueba que contiene un modelo distinto
num_run = 12
z_dim = 4
num_epochs = 20
correr_VAE(vae_dir_model, 'esferas.hdf5', num_run, z_dim, epochs=num_epochs)

# Realizando pruebas con una nueva implementacion, similar a las UNet
# Primera prueba
num_run = 1
z_dim = 4
correr_unet('esferas.hdf5', num_run, z_dim)
z_codes = correr_pruebas_testing(unet_model, 'sph_unet_run_' + str(num_run) +
                                 '.pth', 'esferas.hdf5', 'unet', num_run)

# Segunda prueba
num_run = 2
z_dim = 10
num_epochs = 10
correr_unet('esferas.hdf5', num_run, z_dim, epochs=num_epochs)
z_codes = correr_pruebas_testing(unet_model, 'sph_unet_run_' + str(num_run) +
                                 '.pth', 'esferas.hdf5', 'unet', num_run)

# Tercera prueba
num_run = 3
z_dim = 10
num_epochs = 10
correr_unet('esferas.hdf5', num_run, z_dim, epochs=num_epochs, use_kldiv=False)
z_codes = correr_pruebas_testing(unet_model, 'sph_unet_run_' + str(num_run) +
                                 '.pth', 'esferas.hdf5', 'unet', num_run)

