export PATH="$HOME/Documents/thesis/sdk:$PATH"
set -e
cslc --arch=wse3 ./layout.csl \
  --fabric-dims=512,512 \
  --fabric-offsets=0,0 \
  --params=cells_per_side:64 \
  --params=BITS:6 \
  --params=prob_range_u16:512 \
  --params=max_bodies_pe:2 \
  --params=MEMCPY_H2D_ID:0 \
  --params=MEMCPY_D2H_ID:1 \
  -o out --memcpy --channels 1
#cs_python run.py --name out
