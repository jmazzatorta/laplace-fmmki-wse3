#!/bin/bash
export PATH="$HOME/Documents/thesis/sdk:$PATH"
set -e
cslc --arch=wse3 ./layout.csl \
  --fabric-dims=71,67\
  --fabric-offsets=4,1 \
  --params=max_bodies_pe:5 \
  --params=MEMCPY_H2D_ID:0 \
  --params=MEMCPY_D2H_ID:1 \
  --max-inlined-iterations 500 \
  -o out --memcpy --channels 1

# cs_python run.py --name out
