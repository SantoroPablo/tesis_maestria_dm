#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pablo
"""

# Libraries
import subprocess
import random
import os

# Variables

out_path = 'data/modelos_openscad'
script_path = 'src/openscad/screw_gen_1.scad'

total_outs = 1000 # Numero total de modelos de salida.

random.seed(1)
for i in range(total_outs):
    num           = i+1
    d_ext         = round(random.uniform(10, 25),2)
    thr_step      = random.randint(4,7)
    step_shp_deg  = random.randint(45,55)
    lg_thr_sec    = round(random.uniform(30,60),2)
    cntrsink      = random.randint(1,2)
    hgt_head      = round(random.uniform(8,10),2)
    lg_nonthr_sec = round(random.uniform(2.5,50),2)

    cmd = "openscad -o {out_path}/tornillo_{num}.stl -D \"d_ext={d_ext};"
    cmd = cmd + "thr_step={thr_step};step_shp_deg={step_shp_deg};"
    cmd = cmd + "lg_thr_sec={lg_thr_sec};cntrsink={cntrsink};"
    cmd = cmd + "hgt_head={hgt_head};lg_nonthr_sec={lg_nonthr_sec};\""
    cmd = cmd + " " + script_path
    cmd = cmd.format(out_path=out_path,
                     num=num,
                     d_ext=d_ext,
                     thr_step=thr_step,
                     step_shp_deg=step_shp_deg,
                     lg_thr_sec=lg_thr_sec,
                     cntrsink=cntrsink,
                     hgt_head=hgt_head,
                     lg_nonthr_sec=lg_nonthr_sec)
    os.system(cmd)
