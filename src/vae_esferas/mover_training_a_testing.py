import h5py
import random

data_path = '/home/pablo/org/est/dm/mat/tesis_maestria_dm/repo/data/esferas/esferas.hdf5'

datos = h5py.File(data_path, 'r+')

grp = datos['esferas']

datos.create_group("testing")
datos.move('esferas', 'train')

esferas = [name for name in datos['train']]

random.shuffle(esferas)

# Mando 100 esferas a testing

for i in range(100):
    nombre = esferas[i]
    datos.move('train/' + nombre, 'testing/' + nombre)

datos.close()