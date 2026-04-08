#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder

def interleave_3d(xs, ys, zs, bits=6):
    xs = np.asarray(xs, dtype=np.uint32)
    ys = np.asarray(ys, dtype=np.uint32)
    zs = np.asarray(zs, dtype=np.uint32)
    result = np.zeros(len(xs), dtype=np.uint32)
    for i in range(bits):
        result |= ((xs >> i) & 1) << (3 * i)
        result |= ((ys >> i) & 1) << (3 * i + 1)
        result |= ((zs >> i) & 1) << (3 * i + 2)
    return result

def morton_2d_decode(ids, morton_bits=18):
    ids = np.asarray(ids, dtype=np.uint32)
    t = np.dtype([('row', 'i4'), ('col', 'i4')])
    result = np.zeros(len(ids), dtype=t)
    row_temp = np.zeros(len(ids), dtype=np.uint32)
    col_temp = np.zeros(len(ids), dtype=np.uint32)
    for i in range(morton_bits // 2):
        col_temp |= ((ids >> (i * 2)) & 1) << i
        row_temp |= ((ids >> (i * 2 + 1)) & 1) << i
    result['col'] = col_temp
    result['row'] = row_temp
    return result

parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
    compile_data = json.load(json_file)

# Number of cells per cube side
cells_per_side = int(compile_data['params']['cells_per_side'])

# Physical size of the domain (cube goes from 0 to prob_range)
prob_range = float(compile_data['params']['prob_range'])

# Max bodies per PE
max_bodies_pe = int(compile_data['params']['max_bodies_pe'])

# Number of bodies - hardcoded?
N = 10000

# Validate cube constraint
assert cells_per_side > 0 and (cells_per_side & (cells_per_side - 1)) == 0, \
    f"cells_per_side must be power of 2, got {cells_per_side}"
assert cells_per_side <= 64, \
    f"cells_per_side max 64 (262K PE), got {cells_per_side}"

# Calculate costant values
BITS = int(np.log2(cells_per_side))
TOTAL_CELLS = cells_per_side ** 3
TREE_DEPTH = BITS
MORTON_BITS = 3 * BITS 
GRID_SIDE = 2 ** ((MORTON_BITS + 1) // 2)
CELL_SIZE = prob_range / cells_per_side 

# Display values
print(f"Cube: {cells_per_side}^3 = {TOTAL_CELLS} cells")
print(f"Octree depth: {TREE_DEPTH}")
print(f"Morton bits: {MORTON_BITS}")
print(f"Mesh grid: {GRID_SIDE} x {GRID_SIDE} = {GRID_SIDE**2} PE")
print(f"Cell size: {CELL_SIZE}")
print(f"Bodies: {N}")

# Generate random bodies in 1x1x1 cube with charge q
bodies = np.random.uniform(0, 1, size=(N, 4)).astype(np.float32)

# Assign each body to a cell
cell_coords = (bodies[:, :3] * cells_per_side).astype(np.int32)
cell_coords = np.clip(cell_coords, 0, cells_per_side - 1)

# Calculate Morton ID for each body
unsorted_morton_ids = interleave_3d(cell_coords[:, 0], cell_coords[:, 1], cell_coords[:, 2], BITS)

# Sort bodies by Morton ID 
order = np.argsort(unsorted_morton_ids)
bodies = bodies[order]
body_morton_ids = unsorted_morton_ids[order]

# Group bodies by cell
unique_mortons, start_indices, counts = np.unique(
    body_morton_ids, return_index=True, return_counts=True
)

# Stats
n_occupied = len(unique_mortons)
n_empty = TOTAL_CELLS - n_occupied
max_bodies_in_cell = int(counts.max()) if len(counts) > 0 else 0
# print(f"Occupied cells: {n_occupied} / {TOTAL_CELLS}")
# print(f"Empty cells: {n_empty}")
# print(f"Max bodies per cell: {max_bodies_in_cell}")

assert max_bodies_in_cell <= max_bodies_pe, \
    f"Cell has {max_bodies_in_cell} bodies, exceeds max_bodies_pe={max_bodies_pe}"

# Decode all cells to mesh positions
all_cells_ids = np.arange(TOTAL_CELLS, dtype=np.uint32)
mesh_coords = morton_2d_decode(all_cells_ids, MORTON_BITS)

# Specify path to ELF files, set up runner
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

memcpy_dtype = MemcpyDataType.MEMCPY_32BIT
memcpy_order = MemcpyOrder.ROW_MAJOR

symbol_bodies = runner.get_id("bodies")

runner.load()
runner.run()

print("Copying data...")

# Build PE data
# morton_id (u32) + count (u32) + bodies (max_bodies_pe * 4 floats)
FLOATS_PER_BODY = 4
PER_ELEM_PE = 2 + max_bodies_pe * FLOATS_PER_BODY

# Create flat array for memcpy
total_pe = GRID_SIDE * GRID_SIDE
memcpy_flat = np.zeros(total_pe * PER_ELEM_PE, dtype=np.uint32)

# Build map morton_id -> (start_index, count) in sorted bodies array
cell_map = {}
for i, m in enumerate(unique_mortons):
    cell_map[m] = (start_indices[i], counts[i])

# Fill data for each cell
for m_id in range(TOTAL_CELLS):
    col = int(mesh_coords['col'][m_id])
    row = int(mesh_coords['row'][m_id])
    pe_index = row * GRID_SIDE + col
    offset = pe_index * PER_ELEM_PE

    memcpy_flat[offset] = np.uint32(m_id)

    if m_id in cell_map:
        start, count = cell_map[m_id]
        memcpy_flat[offset + 1] = np.uint32(count)
        for b in range(count):
            body = bodies[start + b]
            base = offset + 2 + b * FLOATS_PER_BODY
            memcpy_flat[base]     = body[0].view(np.uint32)  # x
            memcpy_flat[base + 1] = body[1].view(np.uint32)  # y
            memcpy_flat[base + 2] = body[2].view(np.uint32)  # z
            memcpy_flat[base + 3] = body[3].view(np.uint32)  # q
    else:
        memcpy_flat[offset + 1] = np.uint32(0)

# Copy data to device
runner.memcpy_h2d(
    symbol_bodies, memcpy_flat,
    0, 0, GRID_SIDE, GRID_SIDE, PER_ELEM_PE,
    streaming=False, data_type=memcpy_dtype,
    nonblock=False, order=memcpy_order
)

print("Data copied. Running simulation...")

# runner.memcpy_d2h()
# runner.stop()

