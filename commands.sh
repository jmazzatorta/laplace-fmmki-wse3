#!/bin/bash
export PATH="$HOME/Documents/thesis/sdk:$PATH"
set -e

# Nota: Se cslc si lamenta di risorse di routing ai bordi (tipico del --memcpy), 
# puoi cambiare fabric-dims a 519,514 e fabric-offsets a 4,1. 

cslc --arch=wse3 ./layout.csl \
  --fabric-dims=519,514 \
  --fabric-offsets=4,1 \
  --params=max_bodies_pe:5 \
  --params=MEMCPY_H2D_ID:0 \
  --params=MEMCPY_D2H_ID:1 \
  -o out --memcpy --channels 1

# cs_python run.py --name out
