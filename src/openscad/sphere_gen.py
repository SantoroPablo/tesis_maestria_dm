"""
Generador de esferas de diametro aleatorio entre 0.5 y 1, que se trasladan de
forma aleatoria en un cubo de 3x3x3 (el centro de la esfera).
El objetivo de estos datos de juguete es poder comenzar con las predicciones
en datos 3D para el VAE (Variational Auto Encoder).
"""

# Libraries
import subprocess
import os
import numpy as np
import time

# Constantes
# Ruta al dataset de esferas
OUT_PATH = "/home/pablo/org/est/dm/mat/tesis_maestria_dm/repo/data/"
OUT_PATH = OUT_PATH + "modelos_openscad/esferas/"
NUM_OBJ = 1000
SEED = 0

os.chdir(OUT_PATH)
# Seteo la seed para poder replicar este mismo dataset.
np.random.seed(SEED)
for i in range(NUM_OBJ):
    print('Esfera numero {0:03d}'.format(i))
    x_coord = np.random.uniform(0, 16)
    y_coord = np.random.uniform(0, 16)
    z_coord = np.random.uniform(0, 16)
    # Openscad command string
    # Si el diametro esta entre 8 y 16, el radio esta entre 4 y 8
    radius = np.random.uniform(4, 8)
    str_gen = """
    difference() {{
    cube([32,32,32], center=true, $fn=100);
    translate([{:2f}, {:2f}, {:2f}]) sphere({:2f}, $fn=100);
    }}
    """
    str_gen = str_gen.format(x_coord, y_coord, z_coord, radius)
    archivo = open(OUT_PATH + "temp_script.scad", mode='w')
    archivo.write(str_gen)
    archivo.close()
    cmd = "openscad -o esfera_{0:03d}.stl temp_script.scad"
    cmd = cmd.format(i)
    subprocess.run(cmd.split())
    # Si va demasiado rapido ocurren errores, esto lo ralentiza un poco pero es
    # un tiempo tolerable para generar los 1000 modelos de esferas
    # time.sleep(1)

os.remove("temp_script.scad")

