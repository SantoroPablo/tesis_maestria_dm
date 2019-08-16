"""
Script de eda para las esferas generadas como dataset

Para generar las esferas se usa OpenSCAD junto con python para la iteracion.
"""

# Cargar las librerias para manipular el stl. Probar con PyMesh
from stl import mesh
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import pymesh
import trimesh

# Tomo una esfera de ejemplo
SPH_PATH = '/home/pablo/org/est/dm/mat/tesis_maestria_dm/repo/data/'
SPH_PATH = SPH_PATH + 'modelos_openscad/esferas/train/esfera/esfera_000.stl'

# Visualizacion del stl
sph_mesh = mesh.Mesh.from_file(SPH_PATH)
fig = plt.figure(figsize=(20,10))
ax = mplot3d.Axes3D(fig)

ax.add_collection3d(mplot3d.art3d.Poly3DCollection(sph_mesh.vectors))
scale = sph_mesh.points.flatten(-1)
ax.auto_scale_xyz(scale, scale, scale)
plt.show()

# Voxelizacion del mesh
sph_mesh = trimesh.load(SPH_PATH) # Por alguna razon, tira un warning.
sph_mesh = trimesh.voxel.VoxelMesh(sph_mesh, pitch=1).matrix_solid
print(sph_mesh.shape)
fig = plt.figure(figsize=(20,10))
ax  = fig.gca(projection='3d')
ax.set_aspect('equal')
ax.voxels(sph_mesh, facecolors='y', edgecolors='k')
plt.show()
