#!/usr/bin/env cs_python
import argparse
import json
import pickle
import time
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
    MemcpyDataType,
    MemcpyOrder,
)

N_BODIES       = 10_000
DATA_FILE      = "tables/fmm_full_data.pkl"
M2L_LEVELS     = [2, 3, 4, 5]       # Levels where M2L agendas exist
MAX_AGENDA     = 189                # Max IL size across all levels/PEs
FLOATS_PER_BODY = 4                 # (x, y, z, q)

def interleave_3d(xs, ys, zs, bits):
    xs = np.asarray(xs, dtype=np.uint32)
    ys = np.asarray(ys, dtype=np.uint32)
    zs = np.asarray(zs, dtype=np.uint32)
    result = np.zeros(len(xs), dtype=np.uint32)
    for i in range(bits):
        result |= ((xs >> i) & 1) << (3 * i)
        result |= ((ys >> i) & 1) << (3 * i + 1)
        result |= ((zs >> i) & 1) << (3 * i + 2)
    return result


def morton_2d_decode(ids, morton_bits):
    ids = np.asarray(ids, dtype=np.uint32)
    pe_x = np.zeros(len(ids), dtype=np.uint32)
    pe_y = np.zeros(len(ids), dtype=np.uint32)
    for i in range(morton_bits // 2):
        pe_x |= ((ids >> (i * 2)) & 1) << i
        pe_y |= ((ids >> (i * 2 + 1)) & 1) << i
    return pe_x, pe_y

def build_bodies_tensor(N, cells_per_side, grid_side, max_bodies_pe):
    bits = int(np.log2(cells_per_side))
    morton_bits = 3 * bits
    total_cells = cells_per_side ** 3
    per_elem_pe = 2 + max_bodies_pe * FLOATS_PER_BODY

    # Random bodies in the unit cube; last component is charge
    bodies = np.random.uniform(0.0, 1.0, size=(N, 4)).astype(np.float32)

    # Bin each body into a cubic cell
    cell_coords = (bodies[:, :3] * cells_per_side).astype(np.int32)
    cell_coords = np.clip(cell_coords, 0, cells_per_side - 1)

    # Morton id per body and sort
    body_morton = interleave_3d(cell_coords[:, 0], cell_coords[:, 1], cell_coords[:, 2], bits)
    order = np.argsort(body_morton)
    bodies = bodies[order]
    body_morton = body_morton[order]

    # Group by cell
    unique_mortons, start_idx, counts = np.unique(body_morton, return_index=True, return_counts=True)
    max_cell_load = int(counts.max()) if len(counts) else 0
    assert max_cell_load <= max_bodies_pe, \
        f"A cell has {max_cell_load} bodies, exceeds max_bodies_pe={max_bodies_pe}"

    print(f"  Bodies: {N}  |  Occupied cells: {len(unique_mortons)}/{total_cells}  "
          f"|  Max bodies/cell: {max_cell_load}")

    # Decode cell -> PE coordinates
    all_cells = np.arange(total_cells, dtype=np.uint32)
    cell_pe_x, cell_pe_y = morton_2d_decode(all_cells, morton_bits)

    # Build the flat per-PE layout.
    # Note on ordering: we use ROW_MAJOR on the device side with
    # width=grid_side, height=grid_side. In the flat array a PE at (pe_x, pe_y)
    # lives at index (pe_y * grid_side + pe_x) * per_elem_pe.
    flat = np.zeros(grid_side * grid_side * per_elem_pe, dtype=np.uint32)

    cell_map = {int(m): (int(s), int(c)) for m, s, c in zip(unique_mortons, start_idx, counts)}

    for m_id in range(total_cells):
        px = int(cell_pe_x[m_id])
        py = int(cell_pe_y[m_id])
        base = (py * grid_side + px) * per_elem_pe
        flat[base] = np.uint32(m_id)
        if m_id in cell_map:
            s, c = cell_map[m_id]
            flat[base + 1] = np.uint32(c)
            for b in range(c):
                body = bodies[s + b]
                off = base + 2 + b * FLOATS_PER_BODY
                flat[off]     = body[0].view(np.uint32)
                flat[off + 1] = body[1].view(np.uint32)
                flat[off + 2] = body[2].view(np.uint32)
                flat[off + 3] = body[3].view(np.uint32)
        # else: count stays 0, cell is empty

    return flat, per_elem_pe

def broadcast_array(grid_side, arr):
    n = arr.shape[0]
    return np.broadcast_to(arr, (grid_side, grid_side, n)).copy()


def pack_m2l_agendas(grid_side, pe_schedules, level, max_agenda=MAX_AGENDA):
    agenda_3d = np.zeros((grid_side, grid_side, max_agenda), dtype=np.uint32)
    counts_2d = np.zeros((grid_side, grid_side),             dtype=np.uint32)
    expected_rx_2d = np.zeros((grid_side, grid_side),        dtype=np.uint32)
    for (px, py), sched in pe_schedules.items():
        if level in sched:
            hdrs = sched[level]
            n = len(hdrs)
            assert n <= max_agenda, f"Agenda at ({px},{py}) lvl {level} has {n} > MAX_AGENDA"
            agenda_3d[py, px, :n] = hdrs
            counts_2d[py, px]     = n
            for hdr in hdrs:
                tgt_x = (hdr >> 23) & 0x1FF
                tgt_y = (hdr >> 14) & 0x1FF
                expected_rx_2d[tgt_y, tgt_x] += 1
    return agenda_3d, counts_2d, expected_rx_2d

def h2d_broadcast_u16(runner, symbol, arr_u16, grid_side, elems_per_pe):
    tensor = broadcast_array(grid_side, arr_u16.astype(np.uint16))
    runner.memcpy_h2d(
        symbol, tensor.ravel(),
        0, 0, grid_side, grid_side, elems_per_pe,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_16BIT,
        nonblock=False,
        order=MemcpyOrder.ROW_MAJOR,
    )

def h2d_broadcast_f32(runner, symbol, arr_f32, grid_side, elems_per_pe):
    tensor = broadcast_array(grid_side, arr_f32.astype(np.float32))
    runner.memcpy_h2d(
        symbol, tensor.ravel(),
        0, 0, grid_side, grid_side, elems_per_pe,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
        order=MemcpyOrder.ROW_MAJOR,
    )

def h2d_per_pe_u32(runner, symbol, tensor_3d_or_2d, grid_side, elems_per_pe):
    runner.memcpy_h2d(
        symbol, tensor_3d_or_2d.ravel(),
        0, 0, grid_side, grid_side, elems_per_pe,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False,
        order=MemcpyOrder.ROW_MAJOR,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",   required=True, help="compile output dir")
    parser.add_argument("--cmaddr", help="IP:port for CS system")
    args = parser.parse_args()

    with open(f"{args.name}/out.json", encoding="utf-8") as f:
        compile_data = json.load(f)

    # cells_per_side = int(  compile_data["params"]["cells_per_side"])
    cells_per_side = 16
    # prob_range     = float(compile_data["params"]["prob_range"])
    prob_range = 512
    # max_bodies_pe  = int(  compile_data["params"]["max_bodies_pe"])
    max_bodies_pe  = 5

    assert (cells_per_side & (cells_per_side - 1)) == 0, "cells_per_side must be pow2"
    assert cells_per_side <= 64, "cells_per_side max 64 (262K PE)"

    bits        = int(np.log2(cells_per_side))
    morton_bits = 3 * bits
    grid_side   = 2 ** ((morton_bits + 1) // 2)
    cell_size   = prob_range / cells_per_side

    print("="*60)
    print("FMM-WSE3 host launcher")
    print("="*60)
    print(f"  Cube          : {cells_per_side}^3 = {cells_per_side**3} cells")
    print(f"  Octree depth  : {bits}")
    print(f"  Fabric grid   : {grid_side} x {grid_side} = {grid_side**2} PE")
    print(f"  Cell size     : {cell_size}")
    print(f"  max_bodies_pe : {max_bodies_pe}")

    print(f"\n[1/5] Loading {DATA_FILE} ...")
    with open(DATA_FILE, "rb") as f:
        fmm = pickle.load(f)
    math = fmm["math"]
    pe_schedules = fmm["routing"]["pe_schedules"]
    print(f"      M2M entries : {len(math['m2m']['child_indices'])}")
    print(f"      M2L entries : {len(math['m2l']['source_indices'])}")
    print(f"      PE-nodes    : {len(pe_schedules)}")

    print(f"\n[2/5] Generating {N_BODIES} random bodies and binning ...")
    bodies_flat, per_elem_pe = build_bodies_tensor(
        N_BODIES, cells_per_side, grid_side, max_bodies_pe
    )

    print(f"\n[3/5] Packing per-PE M2L agendas ...")
    agenda_tensors = {}   # level -> (agenda_3d, counts_2d, expected_rx_2d)
    for lvl in M2L_LEVELS:
        a, c, rx = pack_m2l_agendas(grid_side, pe_schedules, lvl)
        agenda_tensors[lvl] = (a, c, rx)
        active = int((c > 0).sum())
        print(f"      lvl {lvl}: active PEs = {active:>6}, max IL = {int(c.max())}")

    print(f"\n[4/5] Loading kernel onto device ...")
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

    runner.load()

    sym = {
        # Bodies buffer (already exists in pe.csl)
        "bodies":             runner.get_id("bodies"),
        # M2M tables
        "m2m_offsets":        runner.get_id("m2m_offsets"),
        "m2m_child_indices":  runner.get_id("m2m_child_indices"),
        "m2m_v_indices":      runner.get_id("m2m_v_indices"),
        "m2m_sign_rr":        runner.get_id("m2m_sign_rr"),
        "m2m_sign_ri":        runner.get_id("m2m_sign_ri"),
        "m2m_sign_ir":        runner.get_id("m2m_sign_ir"),
        "m2m_sign_ii":        runner.get_id("m2m_sign_ii"),
        # M2L tables
        "m2l_offsets":        runner.get_id("m2l_offsets"),
        "m2l_source_indices": runner.get_id("m2l_source_indices"),
        "m2l_v_indices":      runner.get_id("m2l_v_indices"),
        "m2l_sign_rr":        runner.get_id("m2l_sign_rr"),
        "m2l_sign_ri":        runner.get_id("m2l_sign_ri"),
        "m2l_sign_ir":        runner.get_id("m2l_sign_ir"),
        "m2l_sign_ii":        runner.get_id("m2l_sign_ii"),
        # L2L tables
        "l2l_offsets":        runner.get_id("l2l_offsets"),
        "l2l_source_indices": runner.get_id("l2l_source_indices"),
        "l2l_v_indices":      runner.get_id("l2l_v_indices"),
        "l2l_sign_rr":        runner.get_id("l2l_sign_rr"),
        "l2l_sign_ri":        runner.get_id("l2l_sign_ri"),
        "l2l_sign_ir":        runner.get_id("l2l_sign_ir"),
        "l2l_sign_ii":        runner.get_id("l2l_sign_ii"),
        # Output  
        "forces_buf":         runner.get_id("forces_buf"),

    }
    # M2L agendas
    for lvl in M2L_LEVELS:
        sym[f"m2l_agenda_lvl{lvl}"]  = runner.get_id(f"m2l_agenda_lvl{lvl}")
        sym[f"num_targets_lvl{lvl}"] = runner.get_id(f"num_targets_lvl{lvl}")
        sym[f"expected_rx_lvl{lvl}"] = runner.get_id(f"expected_rx_lvl{lvl}")

    print(f"\n[5/5] Pushing data to device ...")
    t0 = time.time()

    m2m = math["m2m"]
    h2d_broadcast_u16(runner, sym["m2m_offsets"],       m2m["offsets"],       grid_side, m2m["offsets"].shape[0])
    h2d_broadcast_u16(runner, sym["m2m_child_indices"], m2m["child_indices"], grid_side, m2m["child_indices"].shape[0])
    h2d_broadcast_u16(runner, sym["m2m_v_indices"],     m2m["v_indices"],     grid_side, m2m["v_indices"].shape[0])
    h2d_broadcast_f32(runner, sym["m2m_sign_rr"],       m2m["sign_rr"],       grid_side, m2m["sign_rr"].shape[0])
    h2d_broadcast_f32(runner, sym["m2m_sign_ri"],       m2m["sign_ri"],       grid_side, m2m["sign_ri"].shape[0])
    h2d_broadcast_f32(runner, sym["m2m_sign_ir"],       m2m["sign_ir"],       grid_side, m2m["sign_ir"].shape[0])
    h2d_broadcast_f32(runner, sym["m2m_sign_ii"],       m2m["sign_ii"],       grid_side, m2m["sign_ii"].shape[0])

    m2l = math["m2l"]
    h2d_broadcast_u16(runner, sym["m2l_offsets"],        m2l["offsets"],        grid_side, m2l["offsets"].shape[0])
    h2d_broadcast_u16(runner, sym["m2l_source_indices"], m2l["source_indices"], grid_side, m2l["source_indices"].shape[0])
    h2d_broadcast_u16(runner, sym["m2l_v_indices"],      m2l["v_indices"],      grid_side, m2l["v_indices"].shape[0])
    h2d_broadcast_f32(runner, sym["m2l_sign_rr"],        m2l["sign_rr"],        grid_side, m2l["sign_rr"].shape[0])
    h2d_broadcast_f32(runner, sym["m2l_sign_ri"],        m2l["sign_ri"],        grid_side, m2l["sign_ri"].shape[0])
    h2d_broadcast_f32(runner, sym["m2l_sign_ir"],        m2l["sign_ir"],        grid_side, m2l["sign_ir"].shape[0])
    h2d_broadcast_f32(runner, sym["m2l_sign_ii"],        m2l["sign_ii"],        grid_side, m2l["sign_ii"].shape[0])

    l2l = math["l2l"]
    h2d_broadcast_u16(runner, sym["l2l_offsets"],        l2l["offsets"],        grid_side, l2l["offsets"].shape[0])
    h2d_broadcast_u16(runner, sym["l2l_source_indices"], l2l["source_indices"], grid_side, l2l["source_indices"].shape[0])
    h2d_broadcast_u16(runner, sym["l2l_v_indices"],      l2l["v_indices"],      grid_side, l2l["v_indices"].shape[0])
    h2d_broadcast_f32(runner, sym["l2l_sign_rr"],        l2l["sign_rr"],        grid_side, l2l["sign_rr"].shape[0])
    h2d_broadcast_f32(runner, sym["l2l_sign_ri"],        l2l["sign_ri"],        grid_side, l2l["sign_ri"].shape[0])
    h2d_broadcast_f32(runner, sym["l2l_sign_ir"],        l2l["sign_ir"],        grid_side, l2l["sign_ir"].shape[0])
    h2d_broadcast_f32(runner, sym["l2l_sign_ii"],        l2l["sign_ii"],        grid_side, l2l["sign_ii"].shape[0])

    for lvl in M2L_LEVELS:
        agenda_3d, counts_2d, expected_rx_2d = agenda_tensors[lvl]
        h2d_per_pe_u32(runner, sym[f"m2l_agenda_lvl{lvl}"],  agenda_3d, grid_side, MAX_AGENDA)
        h2d_per_pe_u32(runner, sym[f"num_targets_lvl{lvl}"], counts_2d, grid_side, 1)
        h2d_per_pe_u32(runner, sym[f"expected_rx_lvl{lvl}"], expected_rx_2d, grid_side, 1)

    h2d_per_pe_u32(runner, sym["bodies"],
                   bodies_flat.reshape(grid_side, grid_side, per_elem_pe),
                   grid_side, per_elem_pe)

    t1 = time.time()
    print(f"      Data push wall-clock : {t1 - t0:.2f} s")

    print(f"\nLaunching upward_phase ...")
    t0 = time.time()
    runner.launch("start_fmm", nonblock=False)
    t1 = time.time()
    print(f"      upward_phase wall-clock : {t1 - t0:.3f} s")

    print(f"\n[6/6] Retrieving forces_buf from device ...")
    t0 = time.time()
    
    forces_out = np.zeros(grid_side * grid_side * max_bodies_pe * FLOATS_PER_BODY, dtype=np.float32)
    
    runner.memcpy_d2h(
        forces_out, sym["forces_buf"],
        0, 0, grid_side, grid_side, max_bodies_pe * FLOATS_PER_BODY,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
        nonblock=False, 
        order=MemcpyOrder.ROW_MAJOR
    )
    
    t1 = time.time()
    print(f"      Data pull wall-clock : {t1 - t0:.2f} s")
    
    forces_3d = forces_out.reshape(grid_side, grid_side, max_bodies_pe, FLOATS_PER_BODY)
    
    total_forces_computed = np.count_nonzero(forces_3d)
    print(f"\nSimulation Complete!")
    print(f"Non-zero output elements retrieved: {total_forces_computed}")

    runner.stop()
    print("\nDone.")

if __name__ == "__main__":
    main()
