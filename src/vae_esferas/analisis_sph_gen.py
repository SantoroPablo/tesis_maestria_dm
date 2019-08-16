import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# TODO: armar ecuacion de la esfera y meterla en una matriz de 32x32x32 en
#       numpy

# Ecuacion de la esfera: https://www.expii.com/t/equation-of-a-sphere-1321

# Coordenadas del centro de la esfera
x = 16
y = 16
z = 16

# Radio de la esfera
r = 16

# La caja contenedora de la esfera es de 32x32x32
bbox = np.zeros((32,32,32), dtype=int)

# Recorro el array para ver si un punto esta dentro o no de la esfera
# Si lo esta, pongo un 1 en la matriz, sino dejo el cero.
for a, yz_ in enumerate(bbox):
    for b, z_ in enumerate(yz):
        for c, _ in enumerate(z_):
            valor_eval = (x-a)**2 + (y-b)**2 + (z-c)**2
            if (valor_eval < r**2):
                bbox[a,b,c] = 1

# Control sobre los valores obtenidos.
# np.sum(bbox)

# Grafico para ver lo que obtuve
fig = plt.figure(figsize=(20,10))
ax = fig.gca(projection='3d')
ax.set_aspect('equal')
ax.voxels(bbox, facecolors='y', edgecolors='k')
plt.show()
