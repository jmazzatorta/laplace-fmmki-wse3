#ifndef WSE3_H
#define WSE3_H

#include <vector>
#include <cmath>

inline int get_idx_wse3(int l, int m, int P) {
    int pos = 0;
    for (int mm = 0; mm < m; mm++) {
        pos += (P + 1 - mm);
    }
    pos += (l - m);
    return pos * 2;
}

inline void compute_solid_harmonics_wse3(double dx, double dy, double dz, double* buf, int P) {
    int buf_size = ((P + 1) * (P + 2) / 2) * 2;
    
    // Clean buffer
    for (int i = 0; i < buf_size; i++) {
        buf[i] = 0.0;
    }

    double r2 = dx * dx + dy * dy + dz * dz;

    // Diagonal {m,m} starting from {0,0}
    double mm_r = 1.0;
    double mm_c = 0.0;

    // Initialize counter for linearized buffer
    int i_m = 0;

    // Outer loop over m
    for (int m = 0; m <= P; m++) {
        
        // Diagonal recurrence
        if (m > 0) {
            double k_diag = -1.0 / (2.0 * m);
            double temp_r = k_diag * (dx * mm_r - dy * mm_c);
            double temp_c = k_diag * (dy * mm_r + dx * mm_c);
            mm_r = temp_r;
            mm_c = temp_c;
        }

        // Store diagonal value
        buf[i_m] = mm_r;
        buf[i_m + 1] = mm_c;
        i_m += 2;

        // Vertical recurrence
        double prev2_r = 0.0;
        double prev2_c = 0.0;
        double prev1_r = mm_r;
        double prev1_c = mm_c;

        for (int l = m + 1; l <= P; l++) {
            
            double k_lm = 1.0 / (l * l - m * m);
            double z_k = (2.0 * l - 1.0) * dz;

            double current_r = k_lm * (z_k * prev1_r - r2 * prev2_r);
            double current_c = k_lm * (z_k * prev1_c - r2 * prev2_c);

            buf[i_m] = current_r;
            buf[i_m + 1] = current_c;
            i_m += 2;

            prev2_r = prev1_r;
            prev2_c = prev1_c;
            prev1_r = current_r;
            prev1_c = current_c;
        }
    }
}

struct M2MTables {
    std::vector<int> displacements;
    std::vector<int> child_indices;
    std::vector<int> v_indices;
    std::vector<double> sign_rr, sign_ri, sign_ir, sign_ii;

    M2MTables(int P) {
        for (int m = 0; m <= P; m++) {
            for (int l = m; l <= P; l++) {
                int count = 0;
                for (int n = 0; n <= P; n++) {
                    for (int k = -n; k <= n; k++) {
                        int c_idx = get_idx_wse3(n, std::abs(k), P);
                        double c_r = (k < 0 && std::abs(k) % 2 != 0) ? -1.0 : 1.0;
                        double c_i = (k < 0) ? -c_r : 1.0;

                        for (int v_l = 0; v_l <= P - n; v_l++) {
                            for (int v_m = -v_l; v_m <= v_l; v_m++) {
                                if (n + v_l == l && k + v_m == m) {
                                    int v_idx = get_idx_wse3(v_l, std::abs(v_m), P);
                                    double v_r = (v_m < 0 && std::abs(v_m) % 2 != 0) ? -1.0 : 1.0;
                                    double v_i = (v_m < 0) ? -v_r : 1.0;

                                    child_indices.push_back(c_idx);
                                    v_indices.push_back(v_idx);
                                    sign_rr.push_back(v_r * c_r);
                                    sign_ri.push_back(-(v_i * c_i));
                                    sign_ir.push_back(v_r * c_i);
                                    sign_ii.push_back(v_i * c_r);
                                    count++;
                                }
                            }
                        }
                    }
                }
                displacements.push_back(count);
            }
        }
    }
};

inline void execute_m2m_wse3(
    double* father_buf, 
    const double* child_buf, 
    const double* V_buf, 
    int P,
    const M2MTables& tables
) {
    int entry_idx = 0;
    int father_logic_idx = 0;

    for (int m = 0; m <= P; m++) {
        for (int l = m; l <= P; l++) {
            
            int count = tables.displacements[father_logic_idx];
            
            int p_idx = father_logic_idx * 2; 

            for (int i = 0; i < count; i++) {
                int c_idx = tables.child_indices[entry_idx];
                int v_idx = tables.v_indices[entry_idx];

                double c_r = child_buf[c_idx];
                double c_i = child_buf[c_idx + 1];

                double v_r = V_buf[v_idx];
                double v_i = V_buf[v_idx + 1];

                father_buf[p_idx]     += (v_r * c_r * tables.sign_rr[entry_idx]) + (v_i * c_i * tables.sign_ri[entry_idx]);
                father_buf[p_idx + 1] += (v_r * c_i * tables.sign_ir[entry_idx]) + (v_i * c_r * tables.sign_ii[entry_idx]);

                entry_idx++;
            }
            father_logic_idx++;
        }
    }
}

#endif // WSE3_H
