import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import h5py

# TODO: armar ecuacion de la esfera y meterla en una matriz de 32x32x32 en
#       numpy

# Ecuacion de la esfera: https://www.expii.com/t/equation-of-a-sphere-1321

# Cantidad de esferas
N = 1000

# hdf5 dataset
f = h5py.File('/home/pablo/org/est/dm/mat/tesis_maestria_dm/repo/data/' +
              'esferas/esferas.hdf5', 'w')
sph_grp = f.create_group("esferas")

np.random.seed(123)

for i in range(N):
    # Coordenadas del centro de la esfera
    x = np.random.uniform(7, 25)
    y = np.random.uniform(7, 25)
    z = np.random.uniform(7, 25)

    centro = np.array([x, y, z])
    # Consideraciones del radio de la esfera con respecto a la bounding box
    r = np.random.uniform(2, 7)

    # La caja contenedora de la esfera es de 32x32x32
    bbox = np.zeros((32, 32, 32), dtype=int)

    # Recorro el array para ver si un punto esta dentro o no de la esfera
    # Si lo esta, pongo un 1 en la matriz, sino dejo el cero.
    for a, yz_ in enumerate(bbox):
        for b, z_ in enumerate(yz_):
            for c, _ in enumerate(z_):
                valor_eval = (x-a)**2 + (y-b)**2 + (z-c)**2
                if valor_eval < r**2:
                    bbox[a, b, c] = 1

    sph_grp.create_dataset("esferas_{0:04d}".format(i), data=bbox)

# Parte en la que tengo que probar si anda h5py
# Cerrando la conexion al archivo
f.close()
