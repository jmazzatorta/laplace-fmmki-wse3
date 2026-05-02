#!/usr/bin/env cs_python
"""
Strada A — diagnostic con flusso intatto ma SENZA tabelle math.
Le tabelle math (m2m, m2l, l2l) sono commentate per ridurre il volume H2D.
I calcoli numerici saranno errati (kernel translate produce 0),
MA il flusso (M2M->M2L->L2L->L2P->P2P->completion) sarà coerente.
Volume H2D ridotto da ~57 MB a ~6 MB. Speedup atteso ~10x.

Cosa viene pushato:
- bodies (necessario per morton_id e count)
- agende M2L (m2l_agenda_lvl{2,3,4}) - necessarie per il flusso M2L
- num_targets_lvl{2,3,4} e expected_rx_lvl{2,3,4} - necessari per i counter
- p2p_agenda + num_p2p_neighbors - necessari per fase P2P

Cosa NON viene pushato (commentato):
- tabelle math m2m, m2l, l2l (offsets, indices, signs)
"""
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

DATA_FILE       = "tables/fmm_full_data.pkl"
MAX_AGENDA      = 189
FLOATS_PER_BODY = 4

TEST_BODIES = np.array([
    [0.1, 0.1, 0.1, 1.0],
    [0.9, 0.9, 0.9, 2.0],
], dtype=np.float32)
N_BODIES = len(TEST_BODIES)


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


def build_bodies_tensor(test_bodies, cells_per_side, grid_side, max_bodies_pe):
    bits = int(np.log2(cells_per_side))
    morton_bits = 3 * bits
    total_cells = cells_per_side ** 3
    per_elem_pe = 2 + max_bodies_pe * FLOATS_PER_BODY

    bodies = test_bodies.copy()
    N = len(bodies)

    cell_coords = (bodies[:, :3] * cells_per_side).astype(np.int32)
    cell_coords = np.clip(cell_coords, 0, cells_per_side - 1)

    body_morton = interleave_3d(cell_coords[:, 0], cell_coords[:, 1], cell_coords[:, 2], bits)
    order = np.argsort(body_morton)
    bodies = bodies[order]
    body_morton = body_morton[order]

    unique_mortons, start_idx, counts = np.unique(body_morton, return_index=True, return_counts=True)
    max_cell_load = int(counts.max()) if len(counts) else 0
    assert max_cell_load <= max_bodies_pe, \
        f"A cell has {max_cell_load} bodies, exceeds max_bodies_pe={max_bodies_pe}"

    print(f"  Bodies: {N}  |  Occupied cells: {len(unique_mortons)}/{total_cells}  "
          f"|  Max bodies/cell: {max_cell_load}")

    all_cells = np.arange(total_cells, dtype=np.uint32)
    cell_pe_x, cell_pe_y = morton_2d_decode(all_cells, morton_bits)

    flat = np.zeros(grid_side * grid_side * per_elem_pe, dtype=np.uint32)
    cell_map = {int(m): (int(s), int(c)) for m, s, c in zip(unique_mortons, start_idx, counts)}

    for m_id in range(total_cells):
        px = int(cell_pe_x[m_id])
        py = int(cell_pe_y[m_id])
        if px >= grid_side or py >= grid_side:
            continue
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

    return flat, per_elem_pe


def broadcast_array(grid_side, arr):
    n = arr.shape[0]
    return np.broadcast_to(arr, (grid_side, grid_side, n)).copy()


def pack_m2l_agendas(grid_side, pe_schedules, level, max_agenda=MAX_AGENDA):
    agenda_3d = np.zeros((grid_side, grid_side, max_agenda), dtype=np.uint32)
    counts_2d = np.zeros((grid_side, grid_side),             dtype=np.uint32)
    expected_rx_2d = np.zeros((grid_side, grid_side),        dtype=np.uint32)
    for (px, py), sched in pe_schedules.items():
        if px >= grid_side or py >= grid_side:
            continue
        if level in sched:
            hdrs = sched[level]
            n = len(hdrs)
            assert n <= max_agenda, f"Agenda at ({px},{py}) lvl {level} has {n} > MAX_AGENDA"
            agenda_3d[py, px, :n] = hdrs
            counts_2d[py, px]     = n
            for hdr in hdrs:
                tgt_x = (hdr >> 23) & 0x1FF
                tgt_y = (hdr >> 14) & 0x1FF
                if tgt_x < grid_side and tgt_y < grid_side:
                    expected_rx_2d[tgt_y, tgt_x] += 1
    return agenda_3d, counts_2d, expected_rx_2d


def pack_p2p_agendas(grid_side, p2p_schedules, max_p2p=26):
    agenda_3d = np.zeros((grid_side, grid_side, max_p2p), dtype=np.uint32)
    counts_2d = np.zeros((grid_side, grid_side), dtype=np.uint32)
    for (px, py), hdrs in p2p_schedules.items():
        if px >= grid_side or py >= grid_side:
            continue
        n = len(hdrs)
        assert n <= max_p2p, f"P2P Agenda at ({px},{py}) has {n} > {max_p2p}"
        agenda_3d[py, px, :n] = hdrs
        counts_2d[py, px] = n
    return agenda_3d, counts_2d


def h2d_broadcast_u16(runner, symbol, arr_u16, grid_side, elems_per_pe):
    if len(arr_u16) % 2 != 0:
        arr_u16 = np.append(arr_u16, [0])
    arr_u32 = np.zeros(len(arr_u16) // 2, dtype=np.uint32)
    arr_u32 = arr_u16[0::2].astype(np.uint32) | (arr_u16[1::2].astype(np.uint32) << 16)
    words_per_pe = len(arr_u32)
    tensor = broadcast_array(grid_side, arr_u32)
    runner.memcpy_h2d(
        symbol, tensor.ravel(),
        0, 0, grid_side, grid_side, words_per_pe,
        streaming=False,
        data_type=MemcpyDataType.MEMCPY_32BIT,
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

    csl_params = compile_data.get("params", {})
    cells_per_side = int(csl_params.get("cells_per_side", 16))
    prob_range     = float(csl_params.get("prob_range", 512.0))
    max_bodies_pe  = int(csl_params.get("max_bodies_pe", 5))

    assert (cells_per_side & (cells_per_side - 1)) == 0, "cells_per_side must be pow2"
    assert cells_per_side <= 64, "cells_per_side max 64 (262K PE)"

    bits        = int(np.log2(cells_per_side))
    morton_bits = 3 * bits
    grid_side   = 2 ** ((morton_bits + 1) // 2)
    cell_size   = prob_range / cells_per_side
    M2L_LEVELS  = list(range(2, bits + 1))

    print("=" * 60)
    print("FMM-WSE3 host launcher  —  STRADA A (no math tables)")
    print("=" * 60)
    print(f"  Cube          : {cells_per_side}^3 = {cells_per_side**3} cells")
    print(f"  Octree depth  : {bits}")
    print(f"  Fabric grid   : {grid_side} x {grid_side} = {grid_side**2} PE")
    print(f"  Cell size     : {cell_size}")
    print(f"  max_bodies_pe : {max_bodies_pe}")
    print(f"  M2L Levels    : {M2L_LEVELS}")
    print()
    print("  ATTENZIONE: Le tabelle math (m2m, m2l, l2l) NON saranno pushate.")
    print("  I calcoli numerici saranno ZERO. Il flusso M2M->M2L->L2L->L2P->P2P")
    print("  girerà correttamente in termini di coordinazione, ma i valori")
    print("  numerici nei coefficienti saranno tutti zero.")
    print("  Test diagnostico: serve solo per verificare il flusso completo.")
    print()

    print(f"[1/5] Loading {DATA_FILE} ...")
    with open(DATA_FILE, "rb") as f:
        fmm = pickle.load(f)
    math = fmm["math"]
    pe_schedules = fmm["routing"]["pe_schedules"]
    p2p_schedules = fmm["routing"]["p2p_schedules"]

    print(f"      M2M entries : {len(math['m2m']['child_indices'])}")
    print(f"      M2L entries : {len(math['m2l']['source_indices'])}")
    print(f"      PE-nodes    : {len(pe_schedules)}")

    print(f"\n[2/5] Loading {N_BODIES} test bodies and binning ...")
    bodies_flat, per_elem_pe = build_bodies_tensor(
        TEST_BODIES, cells_per_side, grid_side, max_bodies_pe
    )

    print(f"\n[3/5] Packing per-PE M2L agendas ...")
    agenda_tensors = {}
    for lvl in M2L_LEVELS:
        a, c, rx = pack_m2l_agendas(grid_side, pe_schedules, lvl)
        agenda_tensors[lvl] = (a, c, rx)
        active = int((c > 0).sum())
        print(f"      lvl {lvl}: active PEs = {active:>6}, max IL = {int(c.max())}")

    print(f"\n[4/5] Loading kernel onto device ...")
    runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
    runner.load()
    runner.run()

    sym = {
        "bodies":             runner.get_id("ptr_bodies"),
        # === COMMENTATO: tabelle math non vengono pushate ===
        # "m2m_offsets":        runner.get_id("ptr_m2m_offsets"),
        # "m2m_child_indices":  runner.get_id("ptr_m2m_child_indices"),
        # "m2m_v_indices":      runner.get_id("ptr_m2m_v_indices"),
        # "m2m_sign_rr":        runner.get_id("ptr_m2m_sign_rr"),
        # "m2m_sign_ri":        runner.get_id("ptr_m2m_sign_ri"),
        # "m2m_sign_ir":        runner.get_id("ptr_m2m_sign_ir"),
        # "m2m_sign_ii":        runner.get_id("ptr_m2m_sign_ii"),
        # "m2l_offsets":        runner.get_id("ptr_m2l_offsets"),
        # "m2l_source_indices": runner.get_id("ptr_m2l_source_indices"),
        # "m2l_v_indices":      runner.get_id("ptr_m2l_v_indices"),
        # "m2l_sign_rr":        runner.get_id("ptr_m2l_sign_rr"),
        # "m2l_sign_ri":        runner.get_id("ptr_m2l_sign_ri"),
        # "m2l_sign_ir":        runner.get_id("ptr_m2l_sign_ir"),
        # "m2l_sign_ii":        runner.get_id("ptr_m2l_sign_ii"),
        # "l2l_offsets":        runner.get_id("ptr_l2l_offsets"),
        # "l2l_source_indices": runner.get_id("ptr_l2l_source_indices"),
        # "l2l_v_indices":      runner.get_id("ptr_l2l_v_indices"),
        # "l2l_sign_rr":        runner.get_id("ptr_l2l_sign_rr"),
        # "l2l_sign_ri":        runner.get_id("ptr_l2l_sign_ri"),
        # "l2l_sign_ir":        runner.get_id("ptr_l2l_sign_ir"),
        # "l2l_sign_ii":        runner.get_id("ptr_l2l_sign_ii"),
        # === FINE COMMENTATO ===
        "forces_buf":         runner.get_id("ptr_forces_buf"),
        "p2p_agenda":         runner.get_id("ptr_p2p_agenda"),
        "num_p2p_neighbors":  runner.get_id("ptr_num_p2p_neighbors"),
    }

    sym["canarino"] = runner.get_id("ptr_canarino")

    for lvl in M2L_LEVELS:
        sym[f"m2l_agenda_lvl{lvl}"]  = runner.get_id(f"ptr_m2l_agenda_lvl{lvl}")
        sym[f"num_targets_lvl{lvl}"] = runner.get_id(f"ptr_num_targets_lvl{lvl}")
        sym[f"expected_rx_lvl{lvl}"] = runner.get_id(f"ptr_expected_rx_lvl{lvl}")

    print(f"\n[5/5] Pushing data to device (NO MATH TABLES) ...")
    t0 = time.time()

    # === COMMENTATO: H2D delle tabelle math ===
    # m2m = math["m2m"]
    # h2d_broadcast_u16(runner, sym["m2m_offsets"],       m2m["offsets"],       grid_side, m2m["offsets"].shape[0])
    # h2d_broadcast_u16(runner, sym["m2m_child_indices"], m2m["child_indices"], grid_side, m2m["child_indices"].shape[0])
    # h2d_broadcast_u16(runner, sym["m2m_v_indices"],     m2m["v_indices"],     grid_side, m2m["v_indices"].shape[0])
    # h2d_broadcast_f32(runner, sym["m2m_sign_rr"],       m2m["sign_rr"],       grid_side, m2m["sign_rr"].shape[0])
    # h2d_broadcast_f32(runner, sym["m2m_sign_ri"],       m2m["sign_ri"],       grid_side, m2m["sign_ri"].shape[0])
    # h2d_broadcast_f32(runner, sym["m2m_sign_ir"],       m2m["sign_ir"],       grid_side, m2m["sign_ir"].shape[0])
    # h2d_broadcast_f32(runner, sym["m2m_sign_ii"],       m2m["sign_ii"],       grid_side, m2m["sign_ii"].shape[0])
    #
    # m2l = math["m2l"]
    # h2d_broadcast_u16(runner, sym["m2l_offsets"],        m2l["offsets"],        grid_side, m2l["offsets"].shape[0])
    # h2d_broadcast_u16(runner, sym["m2l_source_indices"], m2l["source_indices"], grid_side, m2l["source_indices"].shape[0])
    # h2d_broadcast_u16(runner, sym["m2l_v_indices"],      m2l["v_indices"],      grid_side, m2l["v_indices"].shape[0])
    # h2d_broadcast_f32(runner, sym["m2l_sign_rr"],        m2l["sign_rr"],        grid_side, m2l["sign_rr"].shape[0])
    # h2d_broadcast_f32(runner, sym["m2l_sign_ri"],        m2l["sign_ri"],        grid_side, m2l["sign_ri"].shape[0])
    # h2d_broadcast_f32(runner, sym["m2l_sign_ir"],        m2l["sign_ir"],        grid_side, m2l["sign_ir"].shape[0])
    # h2d_broadcast_f32(runner, sym["m2l_sign_ii"],        m2l["sign_ii"],        grid_side, m2l["sign_ii"].shape[0])
    #
    # l2l = math["l2l"]
    # h2d_broadcast_u16(runner, sym["l2l_offsets"],        l2l["offsets"],        grid_side, l2l["offsets"].shape[0])
    # h2d_broadcast_u16(runner, sym["l2l_source_indices"], l2l["source_indices"], grid_side, l2l["source_indices"].shape[0])
    # h2d_broadcast_u16(runner, sym["l2l_v_indices"],      l2l["v_indices"],      grid_side, l2l["v_indices"].shape[0])
    # h2d_broadcast_f32(runner, sym["l2l_sign_rr"],        l2l["sign_rr"],        grid_side, l2l["sign_rr"].shape[0])
    # h2d_broadcast_f32(runner, sym["l2l_sign_ri"],        l2l["sign_ri"],        grid_side, l2l["sign_ri"].shape[0])
    # h2d_broadcast_f32(runner, sym["l2l_sign_ir"],        l2l["sign_ir"],        grid_side, l2l["sign_ir"].shape[0])
    # h2d_broadcast_f32(runner, sym["l2l_sign_ii"],        l2l["sign_ii"],        grid_side, l2l["sign_ii"].shape[0])
    # === FINE COMMENTATO ===

    # H2D necessari per il flusso (agende + counter + bodies)
    for lvl in M2L_LEVELS:
        agenda_3d, counts_2d, expected_rx_2d = agenda_tensors[lvl]
        h2d_per_pe_u32(runner, sym[f"m2l_agenda_lvl{lvl}"],  agenda_3d, grid_side, MAX_AGENDA)
        h2d_per_pe_u32(runner, sym[f"num_targets_lvl{lvl}"], counts_2d, grid_side, 1)
        h2d_per_pe_u32(runner, sym[f"expected_rx_lvl{lvl}"], expected_rx_2d, grid_side, 1)

    p2p_agenda_3d, p2p_counts_2d = pack_p2p_agendas(grid_side, p2p_schedules)
    h2d_per_pe_u32(runner, sym["p2p_agenda"], p2p_agenda_3d, grid_side, 26)
    h2d_per_pe_u32(runner, sym["num_p2p_neighbors"], p2p_counts_2d, grid_side, 1)

    h2d_per_pe_u32(runner, sym["bodies"],
                   bodies_flat.reshape(grid_side, grid_side, per_elem_pe),
                   grid_side, per_elem_pe)

    t1 = time.time()
    print(f"      Data push wall-clock : {t1 - t0:.2f} s")

    print(f"\n[DEBUG] Flushing mem_cpy buffer al device...")
    time.sleep(0.5)

    print(f"\nLaunching upward_phase ...")
    t0 = time.time()
    try:
        runner.launch("start_fmm", nonblock=False)
    except Exception as e:
        print(f"!!! CRASH DURANTE IL LAUNCH !!! Errore: {e}")
    t1 = time.time()
    print(f"      upward_phase wall-clock : {t1 - t0:.3f} s")

    print(f"\n[6/6] Retrieving forces_buf from device ...")

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

    print(f"\n--- RISULTATI DEL SIMULATORE WSE-3 ---")
    print("(Numerici: tutti zero atteso, perché tabelle math non pushate)")
    total_forces_computed = 0
    for py in range(grid_side):
        for px in range(grid_side):
            pe_forces = forces_3d[py, px]
            for i in range(max_bodies_pe):
                fx, fy, fz, pot = pe_forces[i]
                if pot != 0.0 or fx != 0.0 or fy != 0.0 or fz != 0.0:
                    total_forces_computed += 1
                    print(f"PE(X:{px}, Y:{py}) Slot {i} -> F=({fx:.6e}, {fy:.6e}, {fz:.6e}), Pot={pot:.6e}")

    print(f"\nSimulation Complete!")
    print(f"Non-zero output elements retrieved: {total_forces_computed}")

    runner.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
