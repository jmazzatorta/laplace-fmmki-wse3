#!/usr/bin/env python3

import numpy as np
import pickle
import random

P = 4
N_COEFFS   = (P + 1) * (P + 2) // 2
N_COEFFS_V = (2*P + 1) * (2*P + 2) // 2

cells_per_side = 16 
TREE_H = 4 
grid_side = 512 

def get_mem_idx(l, m):
    pos = 0
    for mm in range(m):
        pos += (P + 1 - mm)
    pos += (l - m)
    return pos * 2

def get_v_mem_idx(l, m):
    pos = 0
    for mm in range(m):
        pos += (2*P + 1 - mm)
    pos += (l - m)
    return pos * 2

def parity_sign(k):
    return 1.0 if (abs(k) % 2 == 0) else -1.0

def gen_m2m_tables():
    father_entries = {}
    for l in range(P + 1):
        for m in range(l + 1):
            father_entries[(l, m)] = []

    for n in range(P + 1):
        for k in range(-n, n + 1):
            child_mem_idx = get_mem_idx(n, abs(k))

            if k < 0:
                c_sign_r = parity_sign(k)
                c_sign_i = -parity_sign(k)
            else:
                c_sign_r = 1.0
                c_sign_i = 1.0

            for v_l in range(P - n + 1):
                for v_m in range(-v_l, v_l + 1):
                    l_f = n + v_l
                    m_f = k + v_m

                    if m_f < 0:
                        continue

                    v_mem_idx = get_mem_idx(v_l, abs(v_m))

                    if v_m < 0:
                        v_sign_r = parity_sign(v_m)
                        v_sign_i = -parity_sign(v_m)
                    else:
                        v_sign_r = 1.0
                        v_sign_i = 1.0

                    s_rr = c_sign_r * v_sign_r
                    s_ri = -(c_sign_i * v_sign_i)
                    s_ir = c_sign_r * v_sign_i
                    s_ii = c_sign_i * v_sign_r

                    father_entries[(l_f, m_f)].append({
                        "ci": child_mem_idx,
                        "vi": v_mem_idx,
                        "s_rr": s_rr,
                        "s_ri": s_ri,
                        "s_ir": s_ir,
                        "s_ii": s_ii,
                    })

    offsets, child_indices, v_indices = [], [], []
    sign_rr, sign_ri, sign_ir, sign_ii = [], [], [], []

    curr_offset = 0
    for m in range(P + 1):
        for l in range(m, P + 1):
            entry_list = father_entries[(l, m)]
            offsets.append(curr_offset)

            for entry in entry_list:
                child_indices.append(entry["ci"])
                v_indices.append(entry["vi"])
                sign_rr.append(entry["s_rr"])
                sign_ri.append(entry["s_ri"])
                sign_ir.append(entry["s_ir"])
                sign_ii.append(entry["s_ii"])

            curr_offset += len(entry_list)

    offsets.append(curr_offset)

    return offsets, child_indices, v_indices, sign_rr, sign_ri, sign_ir, sign_ii

def gen_m2l_tables():
    target_entries = {}
    for l in range(P + 1):
        for m in range(l + 1):
            target_entries[(l, m)] = []

    for n in range(P + 1):
        for k in range(-n, n + 1):
            source_mem_idx = get_mem_idx(n, abs(k))

            if k < 0:
                c_sign_r = parity_sign(k)
                c_sign_i = -parity_sign(k)
            else:
                c_sign_r = 1.0
                c_sign_i = 1.0

            for v_l in range(2*P + 1):
                for v_m in range(-v_l, v_l + 1):
                    l_f = v_l - n
                    m_f = k + v_m

                    if l_f < 0 or l_f > P:
                        continue
                    
                    if m_f < 0 or m_f > l_f:
                        continue

                    v_mem_idx = get_v_mem_idx(v_l, abs(v_m))

                    if v_m < 0:
                        v_sign_r = parity_sign(v_m)
                        v_sign_i = -parity_sign(v_m)
                    else:
                        v_sign_r = 1.0
                        v_sign_i = 1.0

                    s_rr = c_sign_r * v_sign_r
                    s_ri = -(c_sign_i * v_sign_i)
                    s_ir = c_sign_r * v_sign_i
                    s_ii = c_sign_i * v_sign_r

                    target_entries[(l_f, m_f)].append({
                        "ci": source_mem_idx,
                        "vi": v_mem_idx,
                        "s_rr": s_rr,
                        "s_ri": s_ri,
                        "s_ir": s_ir,
                        "s_ii": s_ii,
                    })

    offsets, source_indices, v_indices = [], [], []
    sign_rr, sign_ri, sign_ir, sign_ii = [], [], [], []

    curr_offset = 0
    for m in range(P + 1):
        for l in range(m, P + 1):
            entry_list = target_entries[(l, m)]
            offsets.append(curr_offset)

            for entry in entry_list:
                source_indices.append(entry["ci"])
                v_indices.append(entry["vi"])
                sign_rr.append(entry["s_rr"])
                sign_ri.append(entry["s_ri"])
                sign_ir.append(entry["s_ir"])
                sign_ii.append(entry["s_ii"])

            curr_offset += len(entry_list)

    offsets.append(curr_offset)

    return offsets, source_indices, v_indices, sign_rr, sign_ri, sign_ir, sign_ii

def gen_l2l_tables():
    offsets, source_indices, v_indices, sign_rr, sign_ri, sign_ir, sign_ii = gen_m2m_tables()
    return offsets, source_indices, v_indices, sign_rr, sign_ri, sign_ir, sign_ii

def morton_encode_2d(pe_x, pe_y):
    morton_id = 0
    for i in range(9):
        morton_id |= (int(pe_x >> i) & 1) << (i * 2)
        morton_id |= (int(pe_y >> i) & 1) << (i * 2 + 1)
    return morton_id

def morton_decode_2d(morton_id):
    pe_x, pe_y = 0, 0
    for i in range(9):
        pe_x |= ((morton_id >> (2 * i)) & 1) << i
        pe_y |= ((morton_id >> (2 * i + 1)) & 1) << i
    return pe_x, pe_y

def morton_encode_3d(x, y, z):
    m = 0
    for i in range(TREE_H):
        m |= (int(x >> i) & 1) << (3 * i)
        m |= (int(y >> i) & 1) << (3 * i + 1)
        m |= (int(z >> i) & 1) << (3 * i + 2)
    return m

def morton_decode_3d(morton_id):
    x, y, z = 0, 0, 0
    for i in range(TREE_H):
        x |= ((morton_id >> (3 * i)) & 1) << i
        y |= ((morton_id >> (3 * i + 1)) & 1) << i
        z |= ((morton_id >> (3 * i + 2)) & 1) << i
    return x, y, z

def is_pe_node_for_level(pe_x, pe_y, level):
    bits = 3 * (TREE_H - level)
    if bits < 0: return False
    stride_x = 1 << ((bits + 1) // 2)
    stride_y = 1 << (bits // 2)
    return (pe_x % stride_x == 0) and (pe_y % stride_y == 0)

def get_spatial_tag(dx, dy, dz):
    tx, ty, tz = dx + 3, dy + 3, dz + 3
    return tx + (ty * 7) + (tz * 49)

def pack_header(target_x, target_y, level, tag):
    """
    Bit-Packing 32-bit:
    - Target X: 9 bit (23-31)
    - Target Y: 9 bit (14-22)
    - Level:    2 bit (12-13)  [Offset -2]
    - Tag:      9 bit (3-11)   [Range 0-342]
    - Reserved: 3 bit (0-2)
    """
    assert 0 <= target_x <= 511, f"Target X out of bounds: {target_x}"
    assert 0 <= target_y <= 511, f"Target Y out of bounds: {target_y}"
    assert 2 <= level <= TREE_H, f"Level out of bounds: {level}"
    assert 0 <= tag <= 342, f"Tag out of bounds: {tag}"

    header = (target_x & 0x1FF) << 23
    header |= (target_y & 0x1FF) << 14
    header |= ((level - 2) & 0x3) << 12
    header |= (tag & 0x1FF) << 3
    return np.uint32(header)

def gen_routing_agendas():
    pe_schedules = {}
    il_sizes = {l: [] for l in range(2, TREE_H + 1)}
    debug_edges = set() 
    
    for level in range(2, TREE_H + 1):
        cells_per_box = 2**(TREE_H - level)
        grid_size_boxes = cells_per_side // cells_per_box
        
        for box_x in range(grid_size_boxes):
            for box_y in range(grid_size_boxes):
                for box_z in range(grid_size_boxes):
                    
                    s_m3d = morton_encode_3d(box_x * cells_per_box, box_y * cells_per_box, box_z * cells_per_box)
                    s_pe_x, s_pe_y = morton_decode_2d(s_m3d)
                    
                    if not is_pe_node_for_level(s_pe_x, s_pe_y, level): 
                        continue
                        
                    headers = []
                    p_x, p_y, p_z = box_x // 2, box_y // 2, box_z // 2
                    
                    for dpx in [-1, 0, 1]:
                        for dpy in [-1, 0, 1]:
                            for dpz in [-1, 0, 1]:
                                np_x, np_y, np_z = p_x + dpx, p_y + dpy, p_z + dpz
                                
                                if not (0 <= np_x < grid_size_boxes // 2 and 
                                        0 <= np_y < grid_size_boxes // 2 and 
                                        0 <= np_z < grid_size_boxes // 2):
                                    continue
                                
                                for cx in [0, 1]:
                                    for cy in [0, 1]:
                                        for cz in [0, 1]:
                                            t_box_x, t_box_y, t_box_z = np_x * 2 + cx, np_y * 2 + cy, np_z * 2 + cz
                                            
                                            dx = t_box_x - box_x
                                            dy = t_box_y - box_y
                                            dz = t_box_z - box_z
                                            
                                            if max(abs(dx), abs(dy), abs(dz)) > 1:
                                                t_m3d = morton_encode_3d(t_box_x * cells_per_box, t_box_y * cells_per_box, t_box_z * cells_per_box)
                                                t_pe_x, t_pe_y = morton_decode_2d(t_m3d)
                                                
                                                tag = get_spatial_tag(dx, dy, dz)
                                                head = pack_header(t_pe_x, t_pe_y, level, tag)
                                                headers.append(head)
                                                
                                                debug_edges.add((level, (box_x, box_y, box_z), (t_box_x, t_box_y, t_box_z)))

                    il_sizes[level].append(len(headers))
                    if headers:
                        if (s_pe_x, s_pe_y) not in pe_schedules:
                            pe_schedules[(s_pe_x, s_pe_y)] = {}
                        pe_schedules[(s_pe_x, s_pe_y)][level] = np.array(headers, dtype=np.uint32)

    return pe_schedules, il_sizes, debug_edges

def gen_p2p_agendas():
    p2p_schedules = {}
    
    cells_per_box = 1
    grid_size_boxes = cells_per_side  # = 16
    
    for cx in range(grid_size_boxes):
        for cy in range(grid_size_boxes):
            for cz in range(grid_size_boxes):
                s_m3d = morton_encode_3d(cx, cy, cz)
                s_pe_x, s_pe_y = morton_decode_2d(s_m3d)
                
                neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == 0 and dy == 0 and dz == 0:
                                continue  # skip self
                            
                            tx, ty, tz = cx + dx, cy + dy, cz + dz
                            if not (0 <= tx < grid_size_boxes and
                                    0 <= ty < grid_size_boxes and
                                    0 <= tz < grid_size_boxes):
                                continue  # out of bounds
                            
                            t_m3d = morton_encode_3d(tx, ty, tz)
                            t_pe_x, t_pe_y = morton_decode_2d(t_m3d)
                            
                            head = (t_pe_x & 0x1FF) << 23 | (t_pe_y & 0x1FF) << 14
                            neighbors.append(head)
                
                if neighbors:
                    p2p_schedules[(s_pe_x, s_pe_y)] = np.array(neighbors, dtype=np.uint32)
    return p2p_schedules

def gen_binary_blob():
    print("Running Pipeline generator FMM...\n")
    
    print("[1/3] Generating M2M and M2L tables...")
    m2m_out = gen_m2m_tables()
    m2l_out = gen_m2l_tables()
    l2l_out = gen_l2l_tables()
    
    print("[2/3] Generating M2L agendas...")
    pe_schedules, il_sizes, debug_edges = gen_routing_agendas()
    p2p_schedules = gen_p2p_agendas()

    print("\n--- SANITY CHECKS ---")
    # Controllo dinamico sui livelli
    for lvl in range(2, TREE_H + 1):
        sizes = il_sizes[lvl]
        # Previeni crash se un livello non ha agende valide (es. bordi estremi)
        max_sz = max(sizes) if sizes else 0
        avg_sz = (sum(sizes)/len(sizes)) if sizes else 0
        print(f"Level {lvl}: Max IL Size = {max_sz}, Avg = {avg_sz:.1f}")
        
        # L'IL massima (189) si raggiunge con sicurezza dal livello 3 in poi
        if lvl >= 3 and sizes:
            assert max(sizes) == 189, f"Error! Level {lvl} under 189 predicted iterations."

    sym_errors = 0
    # Gestione sicura del random sample se debug_edges è vuoto o piccolo
    sample_size = min(100, len(debug_edges))
    sample_edges = random.sample(list(debug_edges), sample_size) if sample_size > 0 else []
    
    for lvl, src, tgt in sample_edges:
        if (lvl, tgt, src) not in debug_edges: 
            sym_errors += 1
            
    print(f"Test Simmetria IL ({sample_size} random edges): {100 - sym_errors}% passati.")
    if sample_size > 0:
        assert sym_errors == 0, "Errore: La Interaction List NON è perfettamente simmetrica!"
    
    print("\n[3/3] Binary Blob compression...")
    fmm_full_data = {
        "math": {
            "m2m": {
                "offsets":        np.array(m2m_out[0], dtype=np.uint16),
                "child_indices":  np.array(m2m_out[1], dtype=np.uint16),
                "v_indices":      np.array(m2m_out[2], dtype=np.uint16),
                "sign_rr":        np.array(m2m_out[3], dtype=np.float32),
                "sign_ri":        np.array(m2m_out[4], dtype=np.float32),
                "sign_ir":        np.array(m2m_out[5], dtype=np.float32),
                "sign_ii":        np.array(m2m_out[6], dtype=np.float32),
            },
            "m2l": {
                "offsets":        np.array(m2l_out[0], dtype=np.uint16),
                "source_indices": np.array(m2l_out[1], dtype=np.uint16),
                "v_indices":      np.array(m2l_out[2], dtype=np.uint16),
                "sign_rr":        np.array(m2l_out[3], dtype=np.float32),
                "sign_ri":        np.array(m2l_out[4], dtype=np.float32),
                "sign_ir":        np.array(m2l_out[5], dtype=np.float32),
                "sign_ii":        np.array(m2l_out[6], dtype=np.float32),
            },
            "l2l": {
                "offsets":        np.array(l2l_out[0], dtype=np.uint16),
                "source_indices": np.array(l2l_out[1], dtype=np.uint16),
                "v_indices":      np.array(l2l_out[2], dtype=np.uint16),
                "sign_rr":        np.array(l2l_out[3], dtype=np.float32),
                "sign_ri":        np.array(l2l_out[4], dtype=np.float32),
                "sign_ir":        np.array(l2l_out[5], dtype=np.float32),
                "sign_ii":        np.array(l2l_out[6], dtype=np.float32),
            }
        },
        "routing": {
            "pe_schedules": pe_schedules,
            "p2p_schedules": p2p_schedules
        }
    }

    filename = "fmm_full_data.pkl"
    with open(filename, "wb") as f:
        pickle.dump(fmm_full_data, f)
        
    print(f"\n✅ Pipeline succesfully compleated! Exported data in: '{filename}'.")
    print(f"   • P = {P} (N_COEFFS = {N_COEFFS}, N_COEFFS_V = {N_COEFFS_V})")
    print(f"   • Mathematical Entries: M2M = {len(m2m_out[1])}, M2L = {len(m2l_out[1])}")
    print(f"   • PE with Agendas: {len(pe_schedules)}")

if __name__ == "__main__":
    gen_binary_blob()
