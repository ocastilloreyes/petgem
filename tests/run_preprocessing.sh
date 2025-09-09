#!/bin/bash
source ../../set_environment.sh
./gmsh mesh.geo -3
mpirun -n 1 python preprocessing.py
